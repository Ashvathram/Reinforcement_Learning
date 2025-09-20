from flask import Flask, render_template_string, request

app = Flask(__name__)

# Simple HTML template for login form
login_form = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
<form method="post">
  Username: <input type="text" name="username"><br>
  Password: <input type="password" name="password"><br>
  <input type="submit" value="Login">
</form>
{% if message %}<p>{{ message }}</p>{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def login():
    message = ''
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # Simple check (replace with real validation in production)
        if username == 'admin' and password == 'password':
            message = 'Login successful!'
        else:
            message = 'Invalid credentials.'
    return render_template_string(login_form, message=message)

if __name__ == '__main__':
    app.run(debug=True)
