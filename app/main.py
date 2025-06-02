import asyncio
from quart import Quart,request
import sqlite3
from services.parser import FileParser
import os

parser = FileParser("../data/txt.txt")
data = asyncio.run(parser.parse())

print(data)








