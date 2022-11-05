import os
from subprocess import PIPE, run
from paddleocr import VERSION
from dotenv import load_dotenv
import re
from distutils.version import StrictVersion

load_dotenv()

AXIOM_USER = os.environ["AXIOM_USER"]
AXIOM_PASSWORD = os.environ["AXIOM_PASSWORD"]
AXIOM_ID = os.environ["AXIOM_ID"]
AXIOM_VERSION = VERSION
DIST_FILE = f"dist"

successful = False


def run_cmd(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout, result.stderr


def get_list_version(target_string):
    result = re.findall('("name": ")(\d+.\d+.\d+)"', target_string)
    return [r[1] for r in result]


def get_latest_version(target_string):
    list_version = get_list_version(target_string)
    list_version.sort(key=StrictVersion, reverse=True)
    return list_version[0]


stdout, stderr = run_cmd(f"axiom login --email {AXIOM_USER} --password {AXIOM_PASSWORD}")

if "Login successfully" in stdout:
    stdout, stderr = run_cmd(f"axiom model detail --id {AXIOM_ID}")
    latest_version = get_latest_version(stdout)

    print(f"Latest version: {latest_version}")
    print(f"Upload version: {AXIOM_VERSION}")

    if StrictVersion(AXIOM_VERSION) > StrictVersion(latest_version):
        _, _ = run_cmd(f"axiom model tag --id {AXIOM_ID} -v {AXIOM_VERSION}")
        _, _ = run_cmd(f"axiom model upload --id {AXIOM_ID} -v {AXIOM_VERSION} --path {DIST_FILE} --verbose")
        successful = True

    elif StrictVersion(AXIOM_VERSION) == StrictVersion(latest_version):
        _, _ = run_cmd(f"axiom delete tag --id {AXIOM_ID} -v {AXIOM_VERSION}")
        _, _ = run_cmd(f"axiom model upload --id {AXIOM_ID} -v {AXIOM_VERSION} --path {DIST_FILE} --verbose")
        successful = True

    _, _ = run_cmd(f"axiom model set-policy --id {AXIOM_ID} -v {AXIOM_VERSION} --status public")

assert successful
print(f"Upload succesfful version: {AXIOM_VERSION}")