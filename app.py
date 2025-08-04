from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.optimize import linprog
import numpy as np

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

@app.route('/calculate_bwm_weights', methods=['POST'])
def calculate_bwm_weights():
    data = request.json
    criteria_ids = data.get('criteria_ids', [])
    best_criterion_id = data.get('best_criterion_id')
    worst_criterion_id = data.get('worst_criterion_id')
    best_to_others = data.get('best_to_others', {})
    others_to_worst = data.get('others_to_worst', {})

    if not criteria_ids or not best_criterion_id or not worst_criterion_id:
        return jsonify({"error": "Missing criteria information, best, or worst criterion ID."}), 400
    if best_criterion_id not in criteria_ids or worst_criterion_id not in criteria_ids:
        return jsonify({"error": "Best or worst criterion ID not found in criteria list."}), 400
    if best_criterion_id == worst_criterion_id:
        return jsonify({"error": "Best and worst criteria cannot be the same."}), 400

    n = len(criteria_ids)
    if n < 2:
        return jsonify({"error": "BWM requires at least two criteria for comparison."}), 400

    # Map criterion IDs to indices
    id_to_idx = {cid: i for i, cid in enumerate(criteria_ids)}
    best_idx = id_to_idx[best_criterion_id]
    worst_idx = id_to_idx[worst_criterion_id]

    # BWM Linear Programming Formulation
    # Variables: w_1, w_2, ..., w_n, xi (n+1 variables)
    # Objective: Minimize xi (coefficient for xi is 1, others 0)
    # Constraints:
    # 1. |w_B - a_Bj * w_j| <= xi  =>  w_B - a_Bj * w_j - xi <= 0  AND  -w_B + a_Bj * w_j - xi <= 0
    # 2. |w_j - a_jW * w_W| <= xi  =>  w_j - a_jW * w_W - xi <= 0  AND  -w_j + a_jW * w_W - xi <= 0
    # 3. Sum(w_j) = 1
    # 4. w_j >= 0, xi >= 0

    num_vars = n + 1 # n weights + 1 for xi

    # Objective function coefficients (c @ x)
    c = np.zeros(num_vars)
    c[n] = 1 # Minimize xi

    # Inequality constraints (A_ub @ x <= b_ub)
    A_ub = []
    b_ub = []

    # Constraints for Best-to-Others comparisons: |w_B - a_Bj * w_j| <= xi
    for j_id in criteria_ids:
        j_idx = id_to_idx[j_id]
        if j_id != best_criterion_id:
            a_Bj = best_to_others.get(j_id, 1) # Default to 1 if not provided (equal importance)

            # w_B - a_Bj * w_j - xi <= 0
            row1 = np.zeros(num_vars)
            row1[best_idx] = 1
            row1[j_idx] = -a_Bj
            row1[n] = -1 # Coefficient for xi
            A_ub.append(row1)
            b_ub.append(0)

            # -w_B + a_Bj * w_j - xi <= 0
            row2 = np.zeros(num_vars)
            row2[best_idx] = -1
            row2[j_idx] = a_Bj
            row2[n] = -1 # Coefficient for xi
            A_ub.append(row2)
            b_ub.append(0)

    # Constraints for Others-to-Worst comparisons: |w_j - a_jW * w_W| <= xi
    for j_id in criteria_ids:
        j_idx = id_to_idx[j_id]
        if j_id != worst_criterion_id:
            a_jW = others_to_worst.get(j_id, 1) # Default to 1 if not provided

            # w_j - a_jW * w_W - xi <= 0
            row3 = np.zeros(num_vars)
            row3[j_idx] = 1
            row3[worst_idx] = -a_jW
            row3[n] = -1 # Coefficient for xi
            A_ub.append(row3)
            b_ub.append(0)

            # -w_j + a_jW * w_W - xi <= 0
            row4 = np.zeros(num_vars)
            row4[j_idx] = -1
            row4[worst_idx] = a_jW
            row4[n] = -1 # Coefficient for xi
            A_ub.append(row4)
            b_ub.append(0)

    # Equality constraint (A_eq @ x == b_eq)
    # Sum(w_j) = 1
    A_eq = [np.zeros(num_vars)]
    A_eq[0][:n] = 1 # Coefficients for w_j
    b_eq = [1]

    # Bounds for variables (w_j >= 0, xi >= 0)
    # Weights (w_j) must be non-negative. xi (the consistency ratio) must also be non-negative.
    bounds = [(0, None)] * n + [(0, None)] # (0, None) means >= 0

    try:
        # Using 'highs' method for better performance and reliability
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if res.success:
            weights = res.x[:n]
            # Ensure weights sum to exactly 1 due to potential floating point inaccuracies
            sum_of_weights = np.sum(weights)
            if sum_of_weights > 0:
                normalized_weights = weights / sum_of_weights
            else:
                # Fallback in case sum is zero (e.g., all comparisons are 1, leading to all weights being 0 before normalization)
                normalized_weights = np.full(n, 1/n)

            subjective_weights_dict = {
                criteria_ids[i]: float(normalized_weights[i]) for i in range(n)
            }
            return jsonify({"subjective_weights": subjective_weights_dict, "consistency_ratio": float(res.x[n])}), 200
        else:
            return jsonify({"error": f"BWM optimization failed: {res.message}"}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred during BWM calculation: {str(e)}"}), 500

if __name__ == '__main__':
    # For development, run with `flask run`
    # For production, use a WSGI server like Gunicorn
    app.run(debug=True) # debug=True for development, turn off for production
