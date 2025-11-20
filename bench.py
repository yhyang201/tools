import os
import subprocess
import re
import time
import glob
import shutil

def kill_process_by_port(port):
    """Kills any process listening on the specified port."""
    # Try fuser first
    if shutil.which("fuser"):
        subprocess.run(f"fuser -k {port}/tcp", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    
    # Also try lsof just in case
    if shutil.which("lsof"):
        try:
            # Get PID
            cmd = f"lsof -t -i:{port}"
            pid = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
            if pid:
                subprocess.run(f"kill -9 {pid}", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            pass

    # Hard cleanup for sglang serve
    subprocess.run("pkill -f 'sglang serve'", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    
    # Wait a bit for resources to be released
    time.sleep(2)

def run_script(script_path):
    print(f"Running {script_path}...")
    
    # Pre-cleanup if it's an SGL script
    if "sgl/" in script_path:
        kill_process_by_port(30010)

    try:
        # Run the script
        result = subprocess.run(
            ["python", script_path], 
            capture_output=True, 
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        
        if result.returncode != 0:
            print(f"Error running {script_path}:")
            # Print last 20 lines of error
            print("\n".join(result.stderr.splitlines()[-20:]))
            return "Failed"

        # Parse output for Average time
        output = result.stdout
        # Look for "Average time: X.XXXX seconds"
        match = re.search(r"Average time:\s+(\d+\.\d+)\s+seconds", output)
        if match:
            return float(match.group(1))
        else:
            print(f"Could not find average time in output for {script_path}")
            # Debug: print the output if parsing failed
            print("Output snippet:")
            print("\n".join(output.splitlines()[-10:]))
            return "Parse Error"

    except Exception as e:
        print(f"Exception running {script_path}: {e}")
        return "Error"
    finally:
        # Post-cleanup
        if "sgl/" in script_path:
            kill_process_by_port(30010)

def main():
    # Identify all benchmarks by looking at the sgl/ folder
    sgl_scripts = glob.glob("sgl/*.py")
    benchmarks = [os.path.basename(s).replace(".py", "") for s in sgl_scripts]
    benchmarks.sort()

    results = []
    print(f"Found benchmarks: {benchmarks}")
    print("=" * 70)

    for bench in benchmarks:
        row = {"Model": bench}
        
        # 1. Run Diffusers version
        diff_script = f"diffusers/{bench}.py"
        if os.path.exists(diff_script):
            val = run_script(diff_script)
            row["Diffusers (s)"] = val
        else:
            row["Diffusers (s)"] = "N/A"

        # 2. Run SGLang version
        sgl_script = f"sgl/{bench}.py"
        if os.path.exists(sgl_script):
            val = run_script(sgl_script)
            row["SGLang (s)"] = val
        else:
            row["SGLang (s)"] = "N/A"

        # 3. Calculate Speedup
        d_val = row["Diffusers (s)"]
        s_val = row["SGLang (s)"]
        
        if isinstance(d_val, float) and isinstance(s_val, float) and s_val > 0:
             speedup = d_val / s_val
             row["Speedup"] = f"{speedup:.2f}x"
        else:
             row["Speedup"] = "-"

        results.append(row)
        print(f"Finished {bench}: Diffusers={d_val}, SGLang={s_val}")
        print("-" * 70)

    # Print Summary Table
    print("\n\nBenchmark Results Summary:")
    headers = ["Model", "Diffusers (s)", "SGLang (s)", "Speedup"]
    
    # Calculate column widths
    col_widths = [20, 15, 15, 10]
    header_str = f"{headers[0]:<{col_widths[0]}} | {headers[1]:<{col_widths[1]}} | {headers[2]:<{col_widths[2]}} | {headers[3]:<{col_widths[3]}}"
    
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    
    for r in results:
        d_val = f"{r['Diffusers (s)']:.4f}" if isinstance(r['Diffusers (s)'], float) else str(r['Diffusers (s)'])
        s_val = f"{r['SGLang (s)']:.4f}" if isinstance(r['SGLang (s)'], float) else str(r['SGLang (s)'])
        
        row_str = f"{r['Model']:<{col_widths[0]}} | {d_val:<{col_widths[1]}} | {s_val:<{col_widths[2]}} | {r['Speedup']:<{col_widths[3]}}"
        print(row_str)
    
    print("-" * len(header_str))

if __name__ == "__main__":
    main()

