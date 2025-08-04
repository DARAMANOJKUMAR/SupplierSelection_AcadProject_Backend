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
    expert_comparisons = data.get('expert_comparisons', []) # New: expect an array of expert comparisons

    if not criteria_ids or not expert_comparisons:
        return jsonify({"error": "Missing criteria or expert comparison data."}), 400

    n = len(criteria_ids)
    if n < 2:
        return jsonify({"error": "BWM requires at least two criteria for comparison."}), 400

    # --- Step 1: Aggregate Expert Judgments using Geometric Mean ---
    aggregated_best_to_others = {}
    aggregated_others_to_worst = {}

    for j_id in criteria_ids:
        # Aggregate Best-to-Others comparisons for this criterion
        best_to_others_values = [
            exp['best_to_others'].get(j_id, 1) for exp in expert_comparisons if exp['best_criterion_id'] == exp['best_to_others'].get(j_id, 1) or exp['best_criterion_id'] != j_id
        ]
        # In BWM, the comparison of the best criterion to itself is 1. We must filter out non-existent comparisons.
        # This is a robust way to handle the data coming from the frontend, where missing values default to 1.
        
        # Aggregate Others-to-Worst comparisons for this criterion
        others_to_worst_values = [
            exp['others_to_worst'].get(j_id, 1) for exp in expert_comparisons if exp['worst_criterion_id'] == exp['others_to_worst'].get(j_id, 1) or exp['worst_criterion_id'] != j_id
        ]
        
        # Calculate geometric mean
        if best_to_others_values:
            aggregated_best_to_others[j_id] = np.prod(best_to_others_values) ** (1.0 / len(best_to_others_values))
        if others_to_worst_values:
            aggregated_others_to_worst[j_id] = np.prod(others_to_worst_values) ** (1.0 / len(others_to_worst_values))

    # --- Step 2: Identify the overall aggregated Best and Worst Criterion ---
    # Find the criterion with the highest aggregated B-O value (relative to others)
    # The BWM requires the best criterion to be identified *after* aggregation.
    best_id = max(aggregated_best_to_others, key=aggregated_best_to_others.get)
    worst_id = min(aggregated_others_to_worst, key=aggregated_others_to_worst.get)
    
    # BWM optimization requires one best and one worst. 
    # Check if a best and worst was found
    if not best_id or not worst_id:
        return jsonify({"error": "Could not determine an aggregated best or worst criterion from expert data."}), 400

    # Map criterion IDs to indices
    id_to_idx = {cid: i for i, cid in enumerate(criteria_ids)}
    best_idx = id_to_idx[best_id]
    worst_idx = id_to_idx[worst_id]
    
    # --- Step 3: Solve the BWM LP problem with the aggregated data ---
    num_vars = n + 1
    c = np.zeros(num_vars)
    c[n] = 1

    A_ub = []
    b_ub = []

    # Constraints for Best-to-Others comparisons: |w_B - a_Bj * w_j| <= xi
    for j_id in criteria_ids:
        j_idx = id_to_idx[j_id]
        if j_id != best_id:
            a_Bj = aggregated_best_to_others.get(j_id, 1)
            row1 = np.zeros(num_vars)
            row1[best_idx] = 1
            row1[j_idx] = -a_Bj
            row1[n] = -1
            A_ub.append(row1)
            b_ub.append(0)

            row2 = np.zeros(num_vars)
            row2[best_idx] = -1
            row2[j_idx] = a_Bj
            row2[n] = -1
            A_ub.append(row2)
            b_ub.append(0)
    
    # Constraints for Others-to-Worst comparisons: |w_j - a_jW * w_W| <= xi
    for j_id in criteria_ids:
        j_idx = id_to_idx[j_id]
        if j_id != worst_id:
            a_jW = aggregated_others_to_worst.get(j_id, 1)
            row3 = np.zeros(num_vars)
            row3[j_idx] = 1
            row3[worst_idx] = -a_jW
            row3[n] = -1
            A_ub.append(row3)
            b_ub.append(0)
            
            row4 = np.zeros(num_vars)
            row4[j_idx] = -1
            row4[worst_idx] = a_jW
            row4[n] = -1
            A_ub.append(row4)
            b_ub.append(0)

    A_eq = [np.zeros(num_vars)]
    A_eq[0][:n] = 1
    b_eq = [1]
    
    bounds = [(0, None)] * n + [(0, None)]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            weights = res.x[:n]
            sum_of_weights = np.sum(weights)
            if sum_of_weights > 0:
                normalized_weights = weights / sum_of_weights
            else:
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
    app.run(debug=True)
