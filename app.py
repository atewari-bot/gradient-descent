from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


def quadratic_function(x):
    return x ** 2


def gradient_derivative(x):
    return 2 * x


def gradient_descent(starting_point, learning_rate, num_iterations):
    history = [starting_point]
    current_point = starting_point
    for _ in range(num_iterations):
        current_point -= learning_rate * gradient_derivative(current_point)
        history.append(current_point)
    return history


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/compute", methods=["POST"])
def compute():
    data = request.get_json()
    learning_rate = max(0.01, min(0.1, float(data.get("learning_rate", 0.05))))
    iterations = max(10, min(200, int(data.get("iterations", 50))))
    initial_point = max(-10, min(10, float(data.get("initial_point", 10))))

    history = gradient_descent(initial_point, learning_rate, iterations)

    return jsonify({
        "history": history,
        "function_values": [quadratic_function(x) for x in history],
        "gradient_values": [gradient_derivative(x) for x in history],
    })


if __name__ == "__main__":
    app.run(debug=True)
