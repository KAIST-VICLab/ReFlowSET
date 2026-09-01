"""Local-filesystem stand-in for blobfile (no pip in this env)."""
import os, os.path
BlobFile = open
join = os.path.join
dirname = os.path.dirname
basename = os.path.basename
exists = os.path.exists
listdir = os.listdir
isdir = os.path.isdir
