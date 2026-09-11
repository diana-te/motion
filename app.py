import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from solver import solve_motion_captcha

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "online",
        "service": "OnlyFaucet Motion Signature Captcha Solver",
        "version": "1.0",
        "accuracy": "99%+"
    })

@app.route('/solve_motion', methods=['POST'])
@app.route('/solve', methods=['POST'])
def solve_endpoint():
    try:
        data = request.get_json(silent=True) or {}
        raw_image = data.get('image') or request.data.decode('utf-8', errors='ignore')
        
        if not raw_image:
            return jsonify({"status": "error", "message": "No image data provided"}), 400
            
        if ',' in raw_image:
            raw_image = raw_image.split(',', 1)[1]
            
        gif_bytes = base64.b64decode(raw_image)
        p_type, answers = solve_motion_captcha(gif_bytes)
        
        return jsonify({
            "status": "success",
            "motion_type": p_type,
            "answers": answers
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
