import re
import numpy as np
from scipy.io import wavfile
import os

class MuASMCompiler:
    def __init__(self):
        # Initialize registers
        self.registers = {f'R{i}': 0 for i in range(16)}
        # Set default tempo
        self.registers['R1'] = 120  # 120 BPM default
        # Set up sound settings
        self.sample_rate = 44100  # Hz
        self.current_key = 0  # C Major
        self.labels = {}
        self.subroutines = {}
        
        # Wave types (0 = sine, 1 = square, etc.)
        self.wave_types = {
            0: self.generate_sine,
            1: self.generate_square
        }
        self.current_wave = 0  # Default to sine wave
        
        # Instead of playing sounds in real-time, we'll accumulate them
        self.audio_buffer = np.array([], dtype=np.float32)
        
    def parse_file(self, filename):
        """Read and parse a MuASM file and return the resulting audio"""
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                content = file.read()
        else:
            # If file doesn't exist, assume filename is actually the code content
            content = filename
        
        # Reset audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Identify subroutines and labels
        lines = content.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                i += 1
                continue
                
            if line.endswith(':'):
                label_name = line[:-1].strip()
                # Check if it's a subroutine (has content below)
                subroutine_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line or next_line.startswith('#'):
                        j += 1
                        continue
                    if next_line.endswith(':'):
                        break
                    subroutine_lines.append(next_line)
                    j += 1
                self.subroutines[label_name] = subroutine_lines
                i = j
            else:
                i += 1
        
        # Execute the main section if it exists
        if 'main' in self.subroutines:
            self.execute_subroutine('main')
            
        return self.audio_buffer
    
    def execute_subroutine(self, name):
        """Execute a subroutine by name"""
        if name not in self.subroutines:
            print(f"Error: Subroutine '{name}' not found")
            return
        
        lines = self.subroutines[name]
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Handle loop construct
            if line.startswith('loop'):
                match = re.match(r'loop\s+(\d+),\s+(\w+)', line)
                if match:
                    count = int(match.group(1))
                    target = match.group(2)
                    for _ in range(count):
                        self.execute_subroutine(target)
            else:
                self.execute_instruction(line)
            
            i += 1
    
    def execute_instruction(self, instruction):
        """Execute a single MuASM instruction"""
        # Skip empty lines and comments
        if not instruction or instruction.startswith('#'):
            return
            
        parts = instruction.split()
        opcode = parts[0]
        
        if opcode == 'LDI':
            # Load immediate
            # Format: LDI R1, imm
            reg = parts[1].rstrip(',')
            value = int(parts[2])
            self.registers[reg] = value
            
        elif opcode == 'MOV':
            # Move register
            # Format: MOV R1, R2
            dest = parts[1].rstrip(',')
            src = parts[2]
            self.registers[dest] = self.registers[src]
            
        elif opcode == 'ADD':
            # Add
            # Format: ADD R1, R2, R3
            dest = parts[1].rstrip(',')
            src1 = parts[2].rstrip(',')
            src2 = parts[3]
            self.registers[dest] = self.registers[src1] + self.registers[src2]
            
        elif opcode == 'SUB':
            # Subtract
            # Format: SUB R1, R2, R3
            dest = parts[1].rstrip(',')
            src1 = parts[2].rstrip(',')
            src2 = parts[3]
            self.registers[dest] = self.registers[src1] - self.registers[src2]
            
        elif opcode == 'MUL':
            # Multiply
            # Format: MUL R1, R2, R3
            dest = parts[1].rstrip(',')
            src1 = parts[2].rstrip(',')
            src2 = parts[3]
            self.registers[dest] = self.registers[src1] * self.registers[src2]
            
        elif opcode == 'AND':
            # Bitwise AND
            # Format: AND R1, R2, R3
            dest = parts[1].rstrip(',')
            src1 = parts[2].rstrip(',')
            src2 = parts[3]
            self.registers[dest] = self.registers[src1] & self.registers[src2]
            
        elif opcode == 'OR':
            # Bitwise OR
            # Format: OR R1, R2, R3
            dest = parts[1].rstrip(',')
            src1 = parts[2].rstrip(',')
            src2 = parts[3]
            self.registers[dest] = self.registers[src1] | self.registers[src2]
            
        elif opcode == 'BEQ':
            # Branch if equal
            # Format: BEQ R1, R2, label
            reg1 = parts[1].rstrip(',')
            reg2 = parts[2].rstrip(',')
            label = parts[3]
            if self.registers[reg1] == self.registers[reg2]:
                if label in self.labels:
                    return self.labels[label]
                    
        elif opcode == 'BNE':
            # Branch if not equal
            # Format: BNE R1, R2, label
            reg1 = parts[1].rstrip(',')
            reg2 = parts[2].rstrip(',')
            label = parts[3]
            if self.registers[reg1] != self.registers[reg2]:
                if label in self.labels:
                    return self.labels[label]
                    
        elif opcode == 'JMP':
            # Jump
            # Format: JMP label
            label = parts[1]
            if label in self.labels:
                return self.labels[label]
                
        # Music-specific instructions
        elif opcode == 'SETTEMPO':
            # Set tempo
            # Format: SETTEMPO R1
            reg = parts[1]
            self.registers['R1'] = self.registers[reg]
            
        elif opcode == 'SETKEY':
            # Set key
            # Format: SETKEY imm
            value = int(parts[1])
            self.current_key = value
            
        elif opcode == 'PLAYNOTE':
            # Play a note
            # Format: PLAYNOTE R2, R3
            freq_reg = parts[1].rstrip(',')
            dur_reg = parts[2]
            
            frequency = self.registers[freq_reg]
            duration_ms = self.registers[dur_reg]
            
            self.add_note_to_buffer(frequency, duration_ms)
            
        elif opcode == 'TRANSPOSE':
            # Transpose a note
            # Format: TRANSPOSE R2, imm
            reg = parts[1].rstrip(',')
            semitones = int(parts[2])
            
            # Calculate new frequency using the equal temperament formula:
            # new_freq = base_freq * 2^(semitones/12)
            current_freq = self.registers[reg]
            new_freq = current_freq * (2 ** (semitones / 12))
            self.registers[reg] = new_freq
            
        elif opcode == 'TOGGLEWAV':
            # Toggle wave type
            # Format: TOGGLEWAV R5, imm
            reg = parts[1].rstrip(',')
            value = int(parts[2])
            self.registers[reg] = value
            self.current_wave = value
            
        elif opcode == 'STOPNOTE':
            # Stop a note - in our simplified version, we'll just add a short silence
            # Format: STOPNOTE R2
            self.add_silence_to_buffer(100)  # 100ms of silence
            
        elif opcode == 'DIS':
            # Set distortion
            # Format: DIS imm
            value = int(parts[1])
            distortion = max(0, min(10, value))  # Clamp between 0-10
            # In a real implementation, this would modify the sound output
            
        else:
            print(f"Unknown instruction: {opcode}")
    
    def generate_sine(self, frequency, duration_ms):
        """Generate a sine wave"""
        t = np.linspace(0, duration_ms / 1000, int(self.sample_rate * duration_ms / 1000), False)
        wave = np.sin(2 * np.pi * frequency * t) * 0.5
        return wave
    
    def generate_square(self, frequency, duration_ms):
        """Generate a square wave"""
        t = np.linspace(0, duration_ms / 1000, int(self.sample_rate * duration_ms / 1000), False)
        wave = np.sign(np.sin(2 * np.pi * frequency * t)) * 0.5
        return wave
    
    def add_note_to_buffer(self, frequency, duration_ms):
        """Add a note to the audio buffer"""
        # Use the current wave type to generate sound
        wave_generator = self.wave_types.get(self.current_wave, self.generate_sine)
        
        # Generate the wave
        wave = wave_generator(frequency, duration_ms)
        
        # Add to buffer
        self.audio_buffer = np.append(self.audio_buffer, wave)
    
    def add_silence_to_buffer(self, duration_ms):
        """Add silence to the audio buffer"""
        silence = np.zeros(int(self.sample_rate * duration_ms / 1000))
        self.audio_buffer = np.append(self.audio_buffer, silence)
    
    def save_as_wav(self, filename):
        """Save the audio buffer as a WAV file"""
        if len(self.audio_buffer) == 0:
            print("Warning: Audio buffer is empty")
            # Create a short beep as placeholder
            t = np.linspace(0, 1, self.sample_rate)
            self.audio_buffer = np.sin(2 * np.pi * 440 * t) * 0.5
        
        # Normalize audio to prevent clipping
        max_amp = np.max(np.abs(self.audio_buffer))
        if max_amp > 0:
            normalized = self.audio_buffer / max_amp * 0.9
        else:
            normalized = self.audio_buffer
        
        # Convert to 16-bit data
        audio = (normalized * 32767).astype(np.int16)
        
        # Save as WAV
        wavfile.write(filename, self.sample_rate, audio)
        
        return f"Audio saved to {filename}"


def main():
    compiler = MuASMCompiler()
    
    # Example program from the document
    example_program = """
    line: 
    SETTEMPO R1 
    LDI R2, 261 
    LDI R3, 500 
    PLAYNOTE R2, R3 
    TRANSPOSE R2, 4 
    PLAYNOTE R2, R3 
    TRANSPOSE R2, 3 
    PLAYNOTE R2, R3 
    LDI R2, 130 
    LDI R3, 500 
    TOGGLEWAV R5, 1 
    PLAYNOTE R2, R3 
    TRANSPOSE R2, 4 
    PLAYNOTE R2, R3 
    TRANSPOSE R2, 3 
    PLAYNOTE R2, R3 
    TOGGLEWAV R5, 0 
    LDI R2, 523 
    LDI R3, 300 
    PLAYNOTE R2, R3 
    TRANSPOSE R2, 4 
    PLAYNOTE R2, R3 
    TRANSPOSE R2, 3 
    PLAYNOTE R2, R3 
    STOPNOTE R2 
    main: 
    SETKEY 0
    loop 3, line
    SETKEY 4
    loop 3, line
    SETKEY 7
    loop 3, line 
    """
    
    # Save the example to a file
    with open("example.muasm", "w") as f:
        f.write(example_program)
    
    # Parse and execute the program
    compiler.parse_file("example.muasm")
    
    # Save the resulting audio
    compiler.save_as_wav("output.wav")
    
    print("Program executed. Check output.wav for the result.")

if __name__ == "__main__":
    main()
