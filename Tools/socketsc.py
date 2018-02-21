# -*- coding: utf-8 -*-
from __future__ import print_function
import socket


class Socket:
  
  def __init__(self, host='127.0.0.1', port=60000):
    self.host = host
    self.port = port

  def runserver(self):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      sock.bind((self.host, self.port))
      sock.listen(1)

      while True:
        conn, addr = sock.accept()

        with conn:
          while True:
            data = conn.recv(1024)
            if not data:
              break

            print('data: {}, addr: {}'.format(data, addr))
            conn.sendall(b'recieved: ' + data)

  def runclient(self):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      sock.connect((self.host), self.port)
      sock.sendall(b'Hello')
      data = sock.recv(1024)
      print(repr(data))


if __name__ == '__main__':
  
  sock = Socket()
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('mode', choices=['server','client'], help="select server or client.")
  args = parser.parse_args()

  if args.mode == 'server':
    sock.runserver()
  elif args.mode == 'client':
    sock.runclient()
