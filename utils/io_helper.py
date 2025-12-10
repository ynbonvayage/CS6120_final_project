import os

def save_output(method_name, base_name, content):
    out_dir = os.path.join("outputs", base_name)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{base_name}_{method_name}.txt")

    with open(out_path, "w") as f:
        f.write(f"=== {method_name.upper()} VERSION ===\n\n")
        f.write(content)

    print(f"Saved → {out_path}")
