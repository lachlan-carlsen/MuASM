"""
Example usage for the MuASM compiler
"""

from compiler import MuASMCompiler

def main():
    # Create a compiler instance
    compiler = MuASMCompiler()
    
    # Create an example MuASM program
    # This program will play a C major chord arpeggio
    # followed by a G major chord arpeggio
    program = """
    # C Major arpeggio
    c_major:
    LDI R2, 261  # C4 (261.63 Hz)
    LDI R3, 300  # 300ms
    PLAYNOTE R2, R3
    
    LDI R2, 329  # E4 (329.63 Hz)
    PLAYNOTE R2, R3
    
    LDI R2, 392  # G4 (392.00 Hz)
    PLAYNOTE R2, R3
    
    LDI R2, 523  # C5 (523.25 Hz)
    PLAYNOTE R2, R3
    
    # G Major arpeggio
    g_major:
    LDI R2, 392  # G4 (392.00 Hz)
    PLAYNOTE R2, R3
    
    LDI R2, 493  # B4 (493.88 Hz)
    PLAYNOTE R2, R3
    
    LDI R2, 587  # D5 (587.33 Hz)
    PLAYNOTE R2, R3
    
    LDI R2, 784  # G5 (783.99 Hz)
    PLAYNOTE R2, R3
    
    main:
    SETKEY 0  # C Major
    loop 2, c_major
    
    SETKEY 7  # G Major
    loop 2, g_major
    """
    
    # Save the program to a file
    with open("arpeggio.muasm", "w") as f:
        f.write(program)
    
    # Execute the program
    compiler.parse_file("arpeggio.muasm")
    
    # Alternatively, save the output as a WAV file
    compiler.save_program_as_wav("arpeggio.wav", program)
    
    print("Program executed. Check arpeggio.wav for the output.")

if __name__ == "__main__":
    main()
