from http.server import HTTPServer, BaseHTTPRequestHandler
import ssl
import json
import os
import random

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Prepare the JSON response
        # random int between -3 and 3
        level = random.randint(1, 6)
        response_content = json.dumps({"level": level}).encode('utf-8')

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(response_content)  # Sending JSON response

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8443):
    server_address = ('localhost', port)
    httpd = server_class(server_address, handler_class)
    # Wrap the server socket in SSL
    path_to_cert = os.path.join(os.path.dirname(__file__), "cred", "cert.pem")
    path_to_key = os.path.join(os.path.dirname(__file__), "cred", "key.pem")
    httpd.socket = ssl.wrap_socket(httpd.socket,
                                   server_side=True,
                                   certfile=path_to_cert,  # Path to certificate
                                   keyfile=path_to_key,    # Path to private key
                                   ssl_version=ssl.PROTOCOL_TLS)
    print(f"Starting https server on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
