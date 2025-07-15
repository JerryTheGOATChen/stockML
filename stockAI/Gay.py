Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> print("hello World")
hello World
>>> clear
Traceback (most recent call last):
  File "<pyshell#1>", line 1, in <module>
    clear
NameError: name 'clear' is not defined
>>> print("\x1b[2J\x1b[H", end='')
[2J[H
