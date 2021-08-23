# Entry point

import consts
from trainer import Trainer

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os

static_files = [
    '/index.html',
    '/js/viewer.js',
    '/js/chessboard-1.0.0.js',
    '/js/chart.js',
    '/img/chesspieces/wikipedia/bB.png',
    '/img/chesspieces/wikipedia/bK.png',
    '/img/chesspieces/wikipedia/bN.png',
    '/img/chesspieces/wikipedia/bP.png',
    '/img/chesspieces/wikipedia/bQ.png',
    '/img/chesspieces/wikipedia/bR.png',
    '/img/chesspieces/wikipedia/wB.png',
    '/img/chesspieces/wikipedia/wK.png',
    '/img/chesspieces/wikipedia/wN.png',
    '/img/chesspieces/wikipedia/wP.png',
    '/img/chesspieces/wikipedia/wQ.png',
    '/img/chesspieces/wikipedia/wR.png',
    '/css/chessboard-1.0.0.css',
]

staticpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')

trainer = Trainer()
trainer.start_training()

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Default path
        if self.path == '/':
            self.path = '/index.html'

        if self.path in static_files:
            self.send_response(200)

            # Send static ftype header
            if self.path.endswith('html'):
                self.send_header('Content-Type', 'text/html')
            elif self.path.endswith('js'):
                self.send_header('Content-Type', 'application/javascript')
            elif self.path.endswith('css'):
                self.send_header('Content-Type', 'text/css')
            elif self.path.endswith('png'):
                self.send_header('Content-Type', 'image/png')
            
            self.end_headers()
            self.wfile.write(open(os.path.join(staticpath, self.path[1:]), 'rb').read())
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(trainer.status).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

server = HTTPServer(consts.WEBSERVER_BIND, RequestHandler)
print('Listening on %s:%s' % consts.WEBSERVER_BIND)

try:
    server.serve_forever()
except KeyboardInterrupt:
    pass

server.server_close()
print('Stopped server.')