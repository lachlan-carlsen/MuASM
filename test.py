from compiler import MuASMCompiler

compiler = MuASMCompiler()
compiler.parse_file("test.muasm")
compiler.save_as_wav("output.wav")
print("Generated output.wav")
