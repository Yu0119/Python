# -*- coding: utf-8 -*-
"""[Make google search query with urllib]
"""
import webbrowser
import urllib
import urllib.request
import json

payload = { 'q': 'python', 'oq': 'python' }

url = 'https://google.co.jp/search' + '?' + \
        urllib.parse.urlencode(payload)

print(url)
webbrowser.open(url)
