from __future__ import print_function
import numpy as np
import re
import signal
import json
import random
import string
import atexit
import time
from os import listdir, system
from collections import deque
import psutil

#-- Parameters
parsed_registers = ["eax", "ebx", "ecx", "edx", "edi", "esi", "eflags"]
verbose_mode = True
save_extracted_to_folder_path = "extracted"
programs_to_extract_list_path = "food-for-extractor.txt"
check_for_reccuring_patterns = True
max_reccuring_patterns = 10
max_pattern_length = 30
# Software breakpoints are very slow, optimal = 1000?
max_breakpoints = 200
use_debug_symbols = False

attach_to_process = False
process_id = "4316"
program_regex = ""

#-- Helper functions for gdb scripting
# Execute single gdb command
gdbh_exec = lambda s: gdb.execute(s)
# Execute single gdb command returning output to string
gdbh_sexec = lambda s: gdb.execute(s, to_string=True)
# Launch program
gdbh_run = lambda: gdbh_sexec("run")
# Go to main and run
gdbh_start = lambda: gdbh_sexec("start")
# Continue execution
gdbh_continue = lambda: gdbh_sexec("continue")
# Load file into gdb
gdbh_load = lambda path: gdbh_sexec("file " + path)
# Attach gdb to running process
gdbh_attach = lambda pid: gdbh_exec("attach " + pid)
# Step to next instruction
gdbh_step = lambda: gdbh_sexec("ni")
# Get current assembly instruction
gdbh_inst = lambda: gdbh_sexec("x/i $pc")
# Get registers state
gdbh_regs = lambda: gdbh_sexec("info registers")
# Dump first 16 words on stack
gdbh_dump_stack = lambda: gdbh_sexec("x/16xw $esp")
# Exit gdb
gdbh_quit = lambda: gdbh_exec("q")
	
#-- Various helper functions
# Use compiled regex 'rgx' on string 'str' and return all matching groups of first match
def regex_first(rgx, str):
	m = rgx.search(str)
	if m: 
		return m.groups()
	else:
		return None
# Use compiled regex 'rgx' on string 'str' and return all matching groups of all matches
def regex_all(rgx, str):
	m = rgx.findall(str)
	if m: 
		return m
	else:
		return None

def get_random_string(length):
	return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))
	
#-- Convenience class to load programs into gdb and save extracted data
class LoadSaveHandler():
	def __init__(self):
		self.counter = 0
		self.current_program_name = ""
		self.current_program_path = ""
		self.current_function_name = ""
		self.current_save_path = ""
		self.current_function = []
		self.current_sequence = []
		self.current_extract_path = ""
		self.lock_list = []
		self.history = deque([], max_reccuring_patterns * max_pattern_length)
		with open(programs_to_extract_list_path) as f:
			self.programs_to_extract_iterator = iter(f.read().splitlines())
			
	def clear_locks(self):
		print("Cleaning program locks...")
		for lock in self.lock_list:
			try:
				os.remove(lock)
			except:
				pass
	
	def is_program_locked(self, name, create_lock = True):
		lock_path = "./." + name + ".lock"
		if os.path.isfile(lock_path):
			return True
		elif create_lock:
			open(lock_path, "w+")
			self.lock_list.append(lock_path)
		
	def add_to_history(self, fun_name):
		self.history.append(fun_name)
		
	def check_for_reccuring_function_patterns(self):
		if not check_for_reccuring_patterns:
			return False
		min_pattern_length = 2
		correlation = False
		if (len(self.history) < min_pattern_length * max_reccuring_patterns):
			return False
		hist = list(self.history)
		for k in range(min_pattern_length, max_pattern_length+1):
			pattern = hist[-k:]
		
			hist_reduced = hist[-max_reccuring_patterns*k:]
			kernel = pattern * max_reccuring_patterns
			correlation = all([hist_reduced[i] == kernel[i] for i in range(len(kernel))])
			
			if (correlation or max_reccuring_patterns*(k+1) > len(hist)):
				break
		return correlation
	
	# Load a new program to extract into gdb
	def load_new_program(self):		
		if (attach_to_process):
			print("\n###### Attaching to PID " + process_id + " #######")
			print("--> Initialization")
			try:
				gdbh_attach(process_id)
			except:
				print("Can't attach to pid " + process_id)
				raw_input("Press Enter to exit...")				
				quit()
			self.current_program_name = "pid_" + process_id
		else:			
			# Find the next program from the list.txt file
			self.current_program_path = ""
			while True:
				self.current_program_path = next(self.programs_to_extract_iterator)
				self.current_program_name = os.path.splitext(os.path.basename(self.current_program_path))[0]
				if os.path.isfile(self.current_program_path):
					if not self.is_program_locked(self.current_program_name):
						print("\n####### Processing " + self.current_program_name + " #######")
						print("--> Initialization")
						self.current_program_path = '"' + self.current_program_path.replace("\\", "\\\\") + '"'
						break
				else:
					print("/!\ File " + self.current_program_path + " not found.")
			
			gdbh_load(self.current_program_path)
		
		self.current_extract_path = save_extracted_to_folder_path + "/" + self.current_program_name + "/"
		if not os.path.exists(self.current_extract_path):
			os.makedirs(self.current_extract_path)
		
	# Function changed
	def save_new_function(self, instruction):
		# Dump current function to file if not empty
		if self.current_function:
			self.counter = self.counter + 1
			with open(self.current_extract_path + str(self.counter) + "_" + self.current_save_path, "w+") as f:
				json.dump(self.current_function, f)
				self.current_function = []
		
		# Check if the new function has not already been parsed
		file_list = os.listdir(self.current_extract_path)
		self.current_function_name = instruction[1]
		self.add_to_history(self.current_function_name)
		function_name = re.sub('[^\w\-_\. ]', '_', instruction[1])
		if function_name in file_list:
			function_name = function_name + get_random_string(5)
		self.current_save_path = function_name
		
		if verbose_mode:
			print("Entering " + self.current_function_name)
			
	def save_state(self, instruction, regs_before, regs_after, stack_before, stack_after, sequence_changed):
		# No instruction name: we are in the main module
		if (instruction[1] is None):
			if self.current_function_name.startswith(self.current_program_name):
				instruction[1] = self.current_function_name
			else:
				instruction[1] = self.current_program_name
				self.save_new_function(instruction)
		# Else if we stepped in another function
		elif instruction[1] != self.current_function_name:
			self.save_new_function(instruction)
			if (self.check_for_reccuring_function_patterns()):
				print("Repetitive function pattern found. Fast forward...")
				gdbh_continue()
		if (sequence_changed):
			self.current_function.append(self.current_sequence)
			self.current_sequence = []
			if (len(self.current_function) > 10000):
				print("Function is too long, splitting it")
				with open(self.current_extract_path + str(self.counter) + "_" + self.current_save_path, "w+") as f:
					json.dump(self.current_function, f)
				self.current_function = []
				self.counter = self.counter + 1
		self.current_sequence.append([instruction, regs_before, regs_after, stack_before, stack_after])		
		
#-- Core functions
def compile_regexes():
	re_dict = {}
	# pid
	re_dict["pid"] = re.compile(r"^\[New\s+Thread\s+(\d+)", re.MULTILINE)
	# entry point address
	re_dict["init"] = re.compile(r"^\s+Entry\s+point:\s+(0x[0-9a-f]+)", re.MULTILINE)
	# C++ function prototypes in symbols
	re_dict["cpp_func"] = re.compile(r"(?:(?:static )?[a-zA-Z0-9_]+\*? \*?([a-zA-Z0-9_]+)\(.*)+", re.MULTILINE)
	# inst address, func name, offset, hex instruction, asm func name, func arguments
	re_dict["inst"] = re.compile(r"^=>\s+(0x[0-9a-f]+)(?:\s+<([^\+]+)\+(\d+)>){0,1}:\s+((?:[0-9a-f]{2}\s){1,})\s*(\w+)\s+(.*)", re.MULTILINE)
	# register value
	re_dict["regs"] = [re.compile("^" + reg + r"\s+(0x[0-9a-f]*)", re.MULTILINE) for reg in parsed_registers]
	# 16 first words on the stack
	re_dict["stack"] = re.compile("(0x[^:\s]+)(?:\s|$)+", re.MULTILINE)
	# detect unusable instructions
	re_dict["detect_segment"] = re.compile(r"\ws:", re.MULTILINE)
	re_dict["detect_pointer"] = re.compile(r"(PTR|\[|esp)", re.MULTILINE)
	re_dict["detect_callretjmp"] = re.compile(r"(jmp|je|jne|jg|jge|ja|jae|jl|jle|jb|jbe|jo|jno|jz|jnz|js|jns|call|ret)", re.MULTILINE)
	return re_dict

# False if the instruction is messing up with the segments,
# or with pointers, or if it is a call/ret/jmp instruction
def is_instruction_useful(re_dict, parsed_inst):
	if regex_first(re_dict["detect_segment"], parsed_inst[5]) is not None:
		return False
	elif regex_first(re_dict["detect_pointer"], parsed_inst[5]) is not None:
		return False
	elif regex_first(re_dict["detect_callretjmp"], parsed_inst[4]) is not None:
		return False
	else:
		return True
	
def parse_instruction(re_dict, inst_str):
	inst = regex_first(re_dict["inst"], inst_str)
	if inst is not None:
		inst = list(inst) # tuple to list
		inst[3] = inst[3].strip()
		# If unnamed function, then we are in the main module
		return inst			
	else:
		print("Can't parse instruction regex")
	return None

def parse_registers_state(re_dict):
	regs_str = gdbh_regs()
	regs = [regex_first(r, regs_str) for r in re_dict["regs"]]
	return regs
	
def parse_stack_state(re_dict):
	stack_str = gdbh_dump_stack()
	stack_vals = regex_all(re_dict["stack"], stack_str)
	return stack_vals
	
def init_gdb():
	system("cls")
	gdbh_exec("set confirm off")
	gdbh_exec("set disassembly-flavor intel")
	gdbh_exec("set disassemble-next-line on")
	#gdbh_exec("set range-stepping off")

# Kill a process named proc_name
def pkill(proc_name):
    for proc in psutil.process_iter():
        if proc.name() == proc_name:
            proc.kill()

def set_breakpoints_asm(info_functions):
	print("Fetching functions from data sections...")
	func_list = [k.split('  ') for k in info_functions.split('\n')]
	func_list = func_list[6:-1] # 3 first lines are descriptive, last is empty
	addr_list = [fun[0] for fun in func_list]
	if (len(addr_list) > max_breakpoints):
		print("Too many breakpoints (" + str(len(addr_list)) + ")! Downsampling...\n")
	
	step = int(len(addr_list)/max_breakpoints) + 1
	bp_count = 0
	print("--> Placing breakpoints")
	for address in addr_list[::step]:
		try:
			gdbh_sexec("tb *" + address)
			bp_count = bp_count + 1
		except TypeError as e:
			print("Bad address: " + str(address))
			print(e)
			raw_input("Press Enter to exit...")
			quit()
	return bp_count

def set_breakpoints_source(info_functions, re_dict):
	print("Fetching functions from debug symbols...")
	funcs = regex_all(re_dict["cpp_func"], info_functions)
	
	if (len(funcs) > max_breakpoints):
		print("Too many breakpoints (" + str(len(funcs)) + ")! Downsampling...\n")
	
	step = int(len(funcs)/max_breakpoints) + 1
	bp_count = 0
	print("---> Placing breakpoints\n", end="")
	for fun in funcs[::step]:
		print(fun, "- ", end="")
		try:
			gdbh_sexec("tb " + fun)
			bp_count = bp_count + 1
		except TypeError as e:
			print("Bad function: " + str(address))
			print(e)
			raw_input("Press Enter to exit...")
			quit()
	print("")
	return bp_count

def set_breakpoints(loadsave_handler, re_dict):	
	if attach_to_process:
		func_list_str = gdbh_sexec("info functions " + program_regex)
	else:	
		func_list_str = gdbh_sexec("info functions")
		
	if not use_debug_symbols:
		bp_count = set_breakpoints_asm(func_list_str)
	else:
		bp_count = set_breakpoints_source(func_list_str, re_dict)
	
	print(str(bp_count) + " breakpoints set.")
	
# Find entry point, place a breakpoint to it and run
def goto_main_module(re_dict):
	print("Going to main module... ", end="")
	gdbh_start()
	print("Done.\n--> Dumping nasm data")

def next_program(loadsave_handler, re_dict):
	try:
		loadsave_handler.load_new_program()
	except:
		print("No more programs to parse.")
		raw_input("Press Enter to exit...")
		quit()
	finally:
		loadsave_handler.clear_locks()
	
	set_breakpoints(loadsave_handler, re_dict)
	print("--> Running program...")
	gdbh_run()
		
def main():
	re_dict = compile_regexes()
	
	init_gdb()
	loadsave_handler = LoadSaveHandler()
	next_program(loadsave_handler, re_dict)
	
    # all instructions count
	useful_since_last_skip = 0
	useful_total = 0
	useless_since_last_skip = 0
	useless_total = 0
	try:
		sequence_changed = True
		while True:	
			try:
				stack_before = parse_stack_state(re_dict)	
				regs_before = parse_registers_state(re_dict)
			except Exception, e:
				print(e)
				print("Skipping.")
				gdbh_step()
				continue
			inst = gdbh_step()
			inst = parse_instruction(re_dict, inst)
			
			if not inst is None and is_instruction_useful(re_dict, inst):
				stack_after = parse_stack_state(re_dict)	
				regs_after = parse_registers_state(re_dict)
				loadsave_handler.save_state(inst, regs_before, regs_after, stack_before, stack_after, sequence_changed)
				sequence_changed = False
				useful_since_last_skip = useful_since_last_skip + 1
				useful_total = useful_total + 1
			else:
				sequence_changed = True
				useless_since_last_skip = useless_since_last_skip + 1
				useless_total = useless_total + 1
				if useless_total % 1000 == 0:
					print("==> Parsed: " + str(useless_total) + " instructions for " 
					+ str(useful_total) + " useful instructions")
			if useless_since_last_skip > 10000 and useless_since_last_skip > 10*useful_since_last_skip:
				print("\nToo many useless instruction. Skipping...")
				useless_since_last_skip = 0
				useful_since_last_skip = 0
				gdbh_continue()
				
	except Exception, e:
		print(e)
		print("No more programs to parse.")
		raw_input("Press Enter to exit...")
		quit()
	finally:
		loadsave_handler.clear_locks()


if __name__ == "__main__":
	main()

	

	
	
	
