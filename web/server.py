from http.server import SimpleHTTPRequestHandler, HTTPServer, HTTPStatus
from pathlib import Path
import os
import json
import threading

PORT = 6337
IMAGE_PATH = '../data/coco/images/'
IMAGE_FMT  = '%012d.jpg'

class Handler(SimpleHTTPRequestHandler):
	
	def do_GET(self):
		if self.path == '/detindex':
			self.send_str('\n'.join([p.name[:-5] for p in Path('dets/').glob('*.json')]))
		elif self.path.startswith('/image'):
			# Unsafe practices ahead!
			path = self.translate_path(self.path).split('image')
			self.send_file(os.path.join(path[0], IMAGE_PATH, IMAGE_FMT % int(path[1])))
		else:
			super().do_GET()

	def send_str(self, string):
		self.send_response(HTTPStatus.OK)
		self.send_header('Content-type', 'text/plain')
		self.send_header('Content-Length', str(len(string)))
		self.send_header('Last-Modified', self.date_time_string())
		self.end_headers()

		self.wfile.write(string.encode())

	def send_file(self, path):
		try: 
			f = open(path, 'rb')
		except OSError:
			self.send_error(HTTPStatus.NOT_FOUND, "File not found")
			return
		
		try:
			self.send_response(HTTPStatus.OK)
			self.send_header("Content-type", self.guess_type(path))
			fs = os.fstat(f.fileno())
			self.send_header("Content-Length", str(fs[6]))
			self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
			self.end_headers()

			self.copyfile(f, self.wfile)
		finally:
			f.close()

	def send_response(self, code, message=None):
		super().send_response(code, message)

vehicles_list = []
vehicles_mutex = threading.Lock()
class ApiHandler(Handler):
    global vehicles_list
    global vehicles_mutex
    def do_GET(self):
        if self.path == '/api/v1/vehicles.json':
            vehicles_mutex.acquire()
            vehicles_dict = {'vehicles': vehicles_list}
            vehicles_mutex.release()
            self.send_json(json.dumps(vehicles_dict))
        else:
            SimpleHTTPRequestHandler.send_error(self, HTTPStatus.NOT_FOUND, 'cannot found ' + self.path)

    def send_json(self, json_data):
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json_data.encode())

httpd = None

def startHttpServer(request_handler):
    global httpd
    httpd = HTTPServer(('', PORT), request_handler)
    print('Serving at port', PORT)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

def stopHttpServer():
    global httpd
    if httpd is not None:
        print('Stop server')
        httpd.shutdown()

if __name__ == '__main__':
    startHttpServer(Handler)
