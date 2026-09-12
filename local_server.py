import os
import sys
import json
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler

# Pastikan direktori file ini masuk ke sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import solve_motion_captcha

PORT = int(os.environ.get('PORT', 5000))

class MotionSolverHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Requested-With, Authorization')

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self._send_cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        res = {
            "status": "online",
            "service": "OnlyFaucet Motion Local Solver",
            "version": "2.0-RankNet",
            "accuracy": "100% (34/34 dataset)"
        }
        self.wfile.write(json.dumps(res).encode('utf-8'))

    def do_POST(self):
        if self.path not in ('/', '/solve_motion', '/solve'):
            self.send_response(404)
            self._send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": "Not found"}).encode('utf-8'))
            return

        try:
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)

            raw_image = None
            try:
                data = json.loads(body.decode('utf-8'))
                raw_image = data.get('image')
            except Exception:
                raw_image = body.decode('utf-8', errors='ignore')

            if not raw_image:
                self.send_response(400)
                self._send_cors_headers()
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "No image provided"}).encode('utf-8'))
                return

            if ',' in raw_image:
                raw_image = raw_image.split(',', 1)[1]

            gif_bytes = base64.b64decode(raw_image)
            mtype, answers = solve_motion_captcha(gif_bytes)

            self.send_response(200)
            self._send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response_data = {
                "status": "success",
                "motion_type": mtype,
                "answers": answers
            }
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            print(f"[Local Solver] Solved: {mtype} -> Kotak {answers}")

        except Exception as e:
            self.send_response(500)
            self._send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))

    def log_message(self, format, *args):
        pass

def run():
    server_address = ('0.0.0.0', PORT)
    httpd = HTTPServer(server_address, MotionSolverHandler)
    print(f"🚀 OnlyFaucet Local Motion Solver berjalan di http://127.0.0.1:{PORT}")
    print("Tekan Ctrl+C untuk berhenti.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server dihentikan.")
        httpd.server_close()

if __name__ == '__main__':
    run()
