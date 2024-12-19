from flask import Flask, request, jsonify
from optimizer import BayesianOptimizer

app = Flask(__name__)

optimizer = BayesianOptimizer(dimension=3)

@app.route('/observe_user_behavior', methods=['POST'])
def observe_user_behavior():
    global optimizer

    data = request.json
    # chosen_data = data.get('chosen', 0)
    # other_data = data.get('others', 0)
    # for idx in len(chosen_data):    
    #     optimizer.observe_behaivor_estimate(chosen_data[idx], other_data[idx])
    # new_candidates = optimizer.optimize_acqf_and_get_observation().tolist()
    return jsonify({'next_candidates': 'new_candidates'})


@app.route('/upload_text', methods=['POST'])
def upload_text():
    global optimizer

    data = request.json
    # chosen_data = data.get('chosen', 0)
    # other_data = data.get('others', 0)
    # for idx in len(chosen_data):    
    #     optimizer.observe_behaivor_estimate(chosen_data[idx], other_data[idx])
    # new_candidates = optimizer.optimize_acqf_and_get_observation().tolist()
    return jsonify({'upload_text': 'successful'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
