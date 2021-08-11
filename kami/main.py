# Entry point

import consts
from trainer import Trainer

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os

static_files = [
    '/static/index.html'
]

rootpath = os.path.dirname(os.path.dirname(__file__))

trainer = Trainer()
trainer.start_training()

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Default path
        if self.path == '/':
            self.path = '/static/index.html'

        if self.path in static_files:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(open(os.path.join(rootpath, self.path[1:])).read().encode('utf-8'))
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(trainer.status).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

server = HTTPServer(('0.0.0.0', consts.WEB_PORT), RequestHandler)
print('Listening on 0.0.0.0:%s' % consts.WEB_PORT)

try:
    server.serve_forever()
except KeyboardInterrupt:
    pass

server.server_close()
print('Stopped server.')