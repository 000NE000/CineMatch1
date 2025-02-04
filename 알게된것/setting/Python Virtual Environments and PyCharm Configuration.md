# Python Virtual Environments and PyCharm Configuration

## 1. Virtual Environment vs. System-wide Python
| Type                | Description |
|---------------------|-------------|
| **Virtual Environment (venv, Conda)** | Isolated environment for managing dependencies separately for each project. |
| **System-wide Python (/usr/local/bin/python3.x)** | Shared Python installation across all projects, leading to potential dependency conflicts. |

---

## 2. Selecting the Virtual Environment in PyCharm
- **Recommended**: Use the **interpreter from your virtual environment** instead of the system-wide Python.
- Once selected, PyCharm **automatically** applies this interpreter for all project operations.

---

## 3. Installing Packages in a Virtual Environment

### ✅ **Inside PyCharm (No manual activation needed)**
1. **Using PyCharm’s Package Manager**  
   - Go to `File → Settings → Project Interpreter`
   - Click `➕` and install packages via the UI.

2. **Using PyCharm’s Terminal**  
   - Directly run:
     ```sh
     pip install <package>
     ```
     ```sh
     conda install <package>
     ```
   - PyCharm automatically uses the selected virtual environment.

### 🔹 **Outside PyCharm (Manual activation required)**
- In a regular terminal, activate the environment before installing packages:
  ```sh
  conda activate <env_name>