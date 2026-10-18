import os
import json
import re
import time
import uuid
import secrets
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from urllib.parse import urlparse, urljoin

import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, login_required, current_user
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect, CSRFError
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from sqlalchemy import or_
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
IMG_SIZE = 128  # must match IMG_SIZE in train.py

app = Flask(__name__)


# =============================
# Security / App Config
# =============================
def load_secret_key():
    """Use SECRET_KEY env var; otherwise create and reuse a local key file."""
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    key_file = BASE_DIR / ".secret_key"
    if key_file.exists():
        return key_file.read_text().strip()
    key = secrets.token_hex(32)
    key_file.write_text(key)
    try:
        os.chmod(key_file, 0o600)
    except OSError:
        pass
    print("WARNING: SECRET_KEY env var not set. Generated a local .secret_key file. "
          "Set SECRET_KEY in production and never commit .secret_key to git.")
    return key


db_url = os.environ.get("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'users.db').as_posix()}")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config.update(
    SECRET_KEY=load_secret_key(),
    SQLALCHEMY_DATABASE_URI=db_url,
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,                  # 5 MB uploads
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "0") == "1",  # set 1 on HTTPS
    PERMANENT_SESSION_LIFETIME=timedelta(hours=2),
    WTF_CSRF_TIME_LIMIT=None,
)

MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15
RESET_TOKEN_MAX_AGE = 30 * 60  # reset links valid for 30 minutes

db = SQLAlchemy(app)
csrf = CSRFProtect(app)
limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri="memory://")

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.session_protection = "strong"
login_manager.login_message = "Please log in to continue."


def utcnow():
    return datetime.utcnow()


# =============================
# User Model
# =============================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), unique=True, nullable=False, index=True)
    email = db.Column(db.String(254), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    failed_attempts = db.Column(db.Integer, default=0, nullable=False)
    locked_until = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_locked(self):
        return self.locked_until is not None and self.locked_until > utcnow()


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None


with app.app_context():
    db.create_all()

# Used to keep login timing similar whether or not the user exists
DUMMY_HASH = generate_password_hash("dummy-password-for-timing")


# =============================
# Validation Helpers
# =============================
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
COMMON_PASSWORDS = {
    "password", "password1", "password123", "12345678", "123456789", "qwerty123",
    "iloveyou", "admin123", "welcome1", "letmein123", "abc12345", "passw0rd",
}


def validate_password(password, username="", email=""):
    errors = []
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    if len(password) > 128:
        errors.append("Password must be at most 128 characters.")
    if not re.search(r"[a-z]", password):
        errors.append("Password needs a lowercase letter.")
    if not re.search(r"[A-Z]", password):
        errors.append("Password needs an uppercase letter.")
    if not re.search(r"\d", password):
        errors.append("Password needs a number.")
    if password.lower() in COMMON_PASSWORDS:
        errors.append("That password is too common.")
    if username and username.lower() in password.lower():
        errors.append("Password must not contain your username.")
    if email and email.split("@")[0].lower() in password.lower() and len(email.split("@")[0]) >= 4:
        errors.append("Password must not contain your email name.")
    return errors


def is_safe_url(target):
    ref = urlparse(request.host_url)
    test = urlparse(urljoin(request.host_url, target))
    return test.scheme in ("http", "https") and ref.netloc == test.netloc


# =============================
# Password Reset Tokens + Email
# =============================
def _serializer():
    return URLSafeTimedSerializer(app.config["SECRET_KEY"], salt="password-reset")


def make_reset_token(user):
    # Includes a fragment of the current password hash, so the link
    # stops working as soon as the password is changed (single use).
    return _serializer().dumps({"id": user.id, "h": user.password_hash[-12:]})


def verify_reset_token(token):
    try:
        data = _serializer().loads(token, max_age=RESET_TOKEN_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
    user = db.session.get(User, data.get("id"))
    if user and user.password_hash[-12:] == data.get("h"):
        return user
    return None


def send_reset_email(to_email, link):
    host = os.environ.get("MAIL_SERVER")
    if not host:
        # Development fallback: print the link in the console
        print(f"\n[DEV] MAIL_SERVER not set. Password reset link for {to_email}:\n{link}\n")
        return
    msg = EmailMessage()
    msg["Subject"] = "Reset your Malaria AI password"
    msg["From"] = os.environ.get("MAIL_SENDER", os.environ.get("MAIL_USERNAME", ""))
    msg["To"] = to_email
    msg.set_content(
        "We received a request to reset your Malaria AI password.\n\n"
        f"Reset link (valid for {RESET_TOKEN_MAX_AGE // 60} minutes):\n{link}\n\n"
        "If you did not request this, you can ignore this email."
    )
    try:
        with smtplib.SMTP(host, int(os.environ.get("MAIL_PORT", 587)), timeout=15) as smtp:
            smtp.starttls()
            smtp.login(os.environ["MAIL_USERNAME"], os.environ["MAIL_PASSWORD"])
            smtp.send_message(msg)
    except Exception as e:
        app.logger.error("Failed to send reset email: %s", e)


# =============================
# Security Headers
# =============================
@app.after_request
def set_security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    if resp.mimetype == "text/html":
        resp.headers["Cache-Control"] = "no-store"  # no cached pages after logout
    return resp


# =============================
# Auth Routes
# =============================
@app.route("/signup", methods=["GET", "POST"])
@limiter.limit("10 per hour", methods=["POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        errors = []
        if not USERNAME_RE.match(username):
            errors.append("Username must be 3-30 characters: letters, numbers, underscore.")
        if not EMAIL_RE.match(email) or len(email) > 254:
            errors.append("Enter a valid email address.")
        if password != confirm:
            errors.append("Passwords do not match.")
        errors += validate_password(password, username, email)

        if not errors and User.query.filter(
            or_(db.func.lower(User.username) == username.lower(), User.email == email)
        ).first():
            errors.append("Username or email is already in use.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("auth.html", mode="signup", username=username, email=email)

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("auth.html", mode="signup")


@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute", methods=["POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter(
            or_(db.func.lower(User.username) == identifier.lower(), User.email == identifier.lower())
        ).first()

        if user is None:
            check_password_hash(DUMMY_HASH, password)  # equalise timing
        elif not user.is_locked() and user.check_password(password):
            user.failed_attempts = 0
            user.locked_until = None
            db.session.commit()
            session.clear()
            login_user(user, remember=False)
            session.permanent = True
            nxt = request.args.get("next")
            return redirect(nxt if nxt and is_safe_url(nxt) else url_for("home"))
        elif not user.is_locked():
            user.failed_attempts += 1
            if user.failed_attempts >= MAX_FAILED_LOGINS:
                user.locked_until = utcnow() + timedelta(minutes=LOCKOUT_MINUTES)
                user.failed_attempts = 0
            db.session.commit()

        # Same message for wrong password, unknown user, or locked account
        flash(
            f"Invalid username or password. After {MAX_FAILED_LOGINS} failed attempts "
            f"the account is locked for {LOCKOUT_MINUTES} minutes.",
            "error",
        )
        return render_template("auth.html", mode="login", identifier=identifier)

    return render_template("auth.html", mode="login")


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/forgot-password", methods=["GET", "POST"])
@limiter.limit("5 per hour", methods=["POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = User.query.filter_by(email=email).first()
        if user:
            link = url_for("reset_password", token=make_reset_token(user), _external=True)
            send_reset_email(user.email, link)
        # Same response whether or not the account exists (prevents user enumeration)
        flash("If that email is registered, a password reset link has been sent.", "success")
        return redirect(url_for("login"))
    return render_template("auth.html", mode="forgot")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
@limiter.limit("10 per hour", methods=["POST"])
def reset_password(token):
    user = verify_reset_token(token)
    if user is None:
        flash("This reset link is invalid or has expired. Please request a new one.", "error")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        errors = []
        if password != confirm:
            errors.append("Passwords do not match.")
        errors += validate_password(password, user.username, user.email)
        if user.check_password(password):
            errors.append("New password must be different from the old one.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("auth.html", mode="reset", token=token)

        user.set_password(password)
        user.failed_attempts = 0
        user.locked_until = None
        db.session.commit()
        flash("Password updated. You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("auth.html", mode="reset", token=token)


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    flash("Your session expired. Please try again.", "error")
    return redirect(url_for("login"))


@app.errorhandler(413)
def file_too_large(e):
    return render_template("index.html", error="File too large (max 5 MB)"), 413


@app.errorhandler(429)
def too_many_requests(e):
    return "Too many requests. Please wait a few minutes and try again.", 429


# =============================
# Load Model + Class Indices
# =============================
model = tf.keras.models.load_model(BASE_DIR / "malaria_model.keras")

with open(BASE_DIR / "class_indices.json", "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}
print("Class Indices:", class_indices)

infected_index = next(
    (idx for name, idx in class_indices.items() if "parasit" in name.lower()), 0
)
print("Infected class index:", infected_index)

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
UPLOAD_MAX_AGE_SECONDS = 60 * 60
UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.40, 0.60


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_uploads():
    now = time.time()
    for p in UPLOAD_FOLDER.iterdir():
        try:
            if p.is_file() and now - p.stat().st_mtime > UPLOAD_MAX_AGE_SECONDS:
                p.unlink()
        except OSError:
            pass


# =============================
# App Routes (login required)
# =============================
@app.route("/")
@login_required
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@login_required
@limiter.limit("30 per minute")
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected")
    if not allowed_file(file.filename):
        return render_template("index.html", error="Invalid file type (Only PNG/JPG allowed)")

    cleanup_old_uploads()

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4()}_{filename}"
    filepath = UPLOAD_FOLDER / unique_name
    file.save(filepath)

    try:
        img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(image.img_to_array(img), axis=0)
        img_array = preprocess_input(img_array)

        p_index1 = float(model.predict(img_array, verbose=0)[0][0])
        print("Raw Prediction:", p_index1)

        p_infected = p_index1 if infected_index == 1 else 1 - p_index1
        p_uninfected = 1 - p_infected

        if UNCERTAIN_LOW <= p_infected <= UNCERTAIN_HIGH:
            result = "Inconclusive"
            predicted_index = infected_index if p_infected >= 0.5 else 1 - infected_index
            confidence = max(p_infected, p_uninfected) * 100
            recommendation = ("The result is uncertain. Please upload a clearer image "
                              "and consult a doctor or lab for confirmation.")
            status_color = "orange"
        elif p_infected > UNCERTAIN_HIGH:
            result = "Malaria Detected"
            predicted_index = infected_index
            confidence = p_infected * 100
            recommendation = "Malaria detected. Please consult a doctor immediately."
            status_color = "red"
        else:
            result = "Malaria Not Detected"
            predicted_index = 1 - infected_index
            confidence = p_uninfected * 100
            recommendation = "No malaria detected. Maintain hygiene and regular health checkups."
            status_color = "green"

        return render_template(
            "index.html",
            result=result,
            detected_class=index_to_class[predicted_index],
            confidence=f"{confidence:.2f}",
            recommendation=recommendation
            + " (This tool is for screening only, not a medical diagnosis.)",
            image_path=f"static/uploads/{unique_name}",
            status_color=status_color,
        )

    except Exception as e:
        print("Error:", e)
        return render_template("index.html", error="Something went wrong during prediction.")


# =============================
# Run App
# =============================
if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")