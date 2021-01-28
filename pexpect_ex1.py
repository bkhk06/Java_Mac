# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:36:41 2021

@author: Liu.DA
"""

import pexpect
import sys

child = pexpect.spawn('ssh root@192.168.11.102')
fout  = file('mylog.txt','w')
child.logfile = fout
child.logfile = sys.stdout

child.expect("password:")
child.sendline("adccadcc")
child.expect('#')
child.sendline('ls /home')
child.expect('#')
            
             