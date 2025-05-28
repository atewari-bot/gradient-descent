import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def quadratic_function(x):
    # return (x**4 - 15*x**3 + 80*x**2 - 180*x + 144) / 10
    return x**2

def gradient_derivative(x):
    # return (4*x**3 - 45*x**2 + 160*x - 180) / 10
    return 2 * x

st.title("Interactive Gradient Descent Visualization")
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
iterations = st.sidebar.slider("Number of Iterations", min_value=10, max_value=200, value=50, step=5)
initial_point = st.sidebar.number_input("Initial Point", min_value=-10, max_value=10, value=10, step=1)

# Gradient Descent Visualization
def gradient_descent(starting_point, learning_rate, num_iterations):
    history = [starting_point]
    current_point = starting_point

    for _ in range(num_iterations):
        gradient_value = gradient_derivative(current_point) # Calculate the gradient
        current_point -= learning_rate * gradient_value
        history.append(current_point)
    
    return history

def plot_gradient_descent_step_by_step(history):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(-10, 10, 400)
    y = quadratic_function(x)
    plot_placeholder = st.empty()
    plot_placeholder.pyplot(fig)

    for i in range(len(history) - 1):
        ax.clear()
        ax.plot(x, y, label='Quadratic Function $f(x) = x^2$', color='blue')
        ax.set_title('Gradient Descent Step-by-Step')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.axvline(0, color='black', lw=0.5, ls='--')

        # Plot current point
        ax.scatter(history[i], quadratic_function(history[i]), color='red', s=100, label=f'Current Step {i+1}', zorder=5)
        ax.text(history[i], quadratic_function(history[i]), f'{history[i]:.2f}', fontsize=9, ha='right', va='bottom', color='red')
        ax.annotate(f'{history[i - 1]:.2f}', (history[i - 1], quadratic_function(history[i - 1])), textcoords="offset points", xytext=(0,10), ha='center')
        ax.scatter(history[i - 1], quadratic_function(history[i - 1]), color='green', s=100, label=f'Previous Step {i}', zorder=5)

        if i < len(history) - 1:
            next_point = history[i + 1]
            ax.plot([history[i], next_point], [quadratic_function(history[i]), quadratic_function(next_point)], 'r--', alpha=0.5)

        ax.legend()
        plot_placeholder.pyplot(fig)
        time.sleep(0.5)  # Pause for a moment to visualize the step

def main():
    st.write("### Gradient Descent Visualization")
    st.write("This application visualizes the gradient descent algorithm on a simple quadratic function.")
    st.write("You can adjust the learning rate, number of iterations, and initial point using the sidebar.")
    st.write("The red points represent the steps taken by the gradient descent algorithm.")
    st.write("The dashed lines show the path taken from one point to the next.")
    st.write("### Function: $f(x) = x^2$")
    st.write("### Gradient Derivative: $f'(x) = 2x$")
    history = gradient_descent(initial_point, learning_rate, iterations)
    plot_gradient_descent_step_by_step(history)

    st.write("### Gradient Descent Path")
    st.line_chart(pd.Series(history), use_container_width=True)
    st.write("### Final Point")
    st.write(f"Final point after {iterations} iterations: {history[-1]:.2f}")
    st.write("### Function Value at Final Point")
    st.write(f"Function value at final point: {quadratic_function(history[-1]):.2f}")
    st.write("### Gradient at Final Point")
    st.write(f"Gradient at final point: {gradient_derivative(history[-1]):.2f}")

if __name__ == "__main__":
    main()

