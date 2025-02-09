# Python Virtual Environments and PyCharm Configuration

## 왜?
DB 설치는 시스템 레벨에서 실행되며, 전체 OS에서 접근 가능함
- 즉, 가상 환경과 무관하게 OS에서 항상 실행 가능한 서비스가 됨.
- DB는 터미널에서 psql, mongo 같은 명령어를 실행하면 접근할 수 있음

Python 프로젝트는 독립적인 가상 환경에서 실행하면 패키지 충돌을 방지할 수 있음
- Python을 시스템에 직접 설치하면 모든 프로젝트가 하나의 전역 환경을 공유함.
- 프로젝트마다 필요한 패키지 버전이 다를 수 있기 때문에 버전 충돌이 발생할 수 있음
  - 예를 들어, project_A에서는 numpy 1.21.0을 사용하고, project_B에서는 numpy 1.19.0이 필요하면 충돌이 발생함.

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