# Main.py - Server Handling for Neural Network using BaseHTTPRequestHandler
# Jacob Burton 2023

# Constants
ADDRESS = '10.0.0.238'
PORT = 8080

import os

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from twitter_data import *
from model_actions import *

# Http server handling
class HTTPServer_RequestHandler(BaseHTTPRequestHandler):
    # GET Requests
    def do_GET(self):
        # Split Path by / to get route
        route = self.path.split('/')
        #print(route) # debug

        # File Requested
        permitted_extensions = ['.html','.png','.svg','.jpg', '.js', '.json', '.css', '.ico'] # Allowed File types
        file_request = False
        if os.path.splitext(self.path)[1] in permitted_extensions:
            file_request = True

        # Set the path to index.html if the path is empty
        if (self.path == '/') or file_request:
            if (self.path == '/'):
                self.path = '/index.html'
            
            # Try to open the file
            try:
                # Parse Path
                parsed_path = urlparse(self.path)
                file_path = './www/' + parsed_path.path

                # Open the file
                file = open(file_path, 'rb')
            except IOError:
                self.send_error(404, 'File Not Found')
                return
            
            # Send response
            self.send_response(200)
            self.end_headers()

            # Send the file
            self.wfile.write(file.read())
            file.close()
            
            return
        # /tweet/<id> - Get Tweet
        elif route[1] == 'tweet':
            # Get the tweet
            tweet_id = route[2]
            try:
                tweet_data = grab_tweet(tweet_id)
            except Exception as e:
                result = (repr(e).encode('utf-8'))
                self.send_response(404, result)
                self.end_headers()
                return

            # Decode from latin-1 to utf-8
            result = tweet_data.encode('latin-1', 'ignore').decode('utf-8')

            # Send the result  
            self.send_response(200, result)
            self.end_headers()
            return
        # /evaluate/<id> - Evaluate Tweet on the Network
        elif route[1] == 'evaluate':      
            # Load the Model
            model = load_model("models/trained_model1.model")

            # Get the tweet and evaluate it
            tweet_id = route[2]
            try:
                confidence_percent, prediction = evaluate_tweet(model, tweet_id)
            except Exception as e:
                result = (repr(e).encode('utf-8'))
                self.send_response(404, result)
                self.end_headers()
                return

            # Tweet Found, parse result
            result = str(f'{confidence_percent:.3f}%' + "/" + prediction)

            # Send the result
            self.send_response(200, result)
            self.end_headers()
            return
        else:
            self.send_error(404, 'URI Not Found')
            return
        
    # POST Requests
    def do_POST(self):
        # Split Path by / to get route
        route = self.path.split('/')
        print(route) # debug

        # /submit/<id>/<category> - Submit Tweet to be trained in the model
        if route[1] == 'submit':  
            # Get the tweet
            tweet_id = route[2]
            category = route[3]
            try:
                # Submit tweet to Submission Dataset
                result = add_tweet_by_id(tweet_id, category, "tweet_data/submissions/submitted_data.json")
                if result == 1:
                    self.send_response(200, "Success")
                    self.end_headers()
                    return
                else:
                    self.send_response(404, "Error")
                    self.end_headers()
                return
            except Exception as e:
                result = (repr(e).encode('utf-8'))
                self.send_response(404, result)
                self.end_headers()
                return
            
            
            

# Run the server
def run():
    print('Starting server...')
    server_address = (ADDRESS, PORT)
    httpd = HTTPServer(server_address, HTTPServer_RequestHandler)
    print('Running server on port %s' % PORT)

    # Try run server until keyboard interrupt
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

run()