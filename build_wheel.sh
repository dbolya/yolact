#!/usr/bin/env bash

#************************************************#
#           build_wheel.sh                       #
#    Build a Python3 wheel for YOLACT            #
#************************************************#

REPO_NAME="yolact"
NEEDED_FOLDERS=("data" "external" "layers" "scripts" "utils" "web")
REQUIREMENTS_FILE="requirements.txt"
SETUP_FILE="setup.py"
PYTHON_EXTENSION=".py"
PACKAGE_TEST_DIR="/tmp/$USER/pip_package_test"

# --------------------------------------------------------- #
# main ()                                                   #
# drives the entire build process                           #
# Returns: 0 on success, non-zero if something goes wrong.  #
# --------------------------------------------------------- #
main() {
  echo "Starting build ..."
  pre_build_setup &&
    build &&
    post_build_test_and_teardown &&
    echo "Build Successful ... "
}

# --------------------------------------------------------- #
# pre_build_setup ()                                        #
# performs necessary setup for building a YOLACT wheel      #
# Returns: 0 on success, non-zero if something goes wrong.  #
# --------------------------------------------------------- #
pre_build_setup() {
  echo "Pre-Build Setup ... "
  __validate_files && __copy_files && __fix_requirements && __build_setup
}

# --------------------------------------------------------- #
# build ()                                                  #
# actual build happens here                                 #
# Returns: 0 on success, non-zero if something goes wrong.  #
# --------------------------------------------------------- #
build() {
  echo "Build ... "
  python3 -m build
  rm --recursive "${REPO_NAME}"
}

# --------------------------------------------------------- #
# post_build_test_and_teardown ()                           #
# installation check + cleanup unnecessary files            #
# Returns: 0 on success, non-zero if something goes wrong.  #
# --------------------------------------------------------- #
post_build_test_and_teardown() {
  echo "Post-Build Test and Teardowns ... "
  __test_build
  __teardown
}

__validate_files() {
  echo "Validating files ... "
  # venv
  if [[ ! -e "$(command -v python3)" ]]; then
    echo "Is python installed?" 1>&2
    return 1
  fi

  # setup.py
  if [[ ! -e "${SETUP_FILE}" ]]; then
    echo "${SETUP_FILE} file not found" 1>&2
    return 3
  fi
}

__copy_files() {
  echo "Copying files ... "

  # create empty yolact dir, overwrite if exists
  [[ -d ${REPO_NAME} ]] && rm --recursive ${REPO_NAME}
  mkdir --parents "${REPO_NAME}"

  for folder in "${NEEDED_FOLDERS[@]}"; do
    cp --recursive "${folder}" "${REPO_NAME}"
  done

  echo "Copying ${REQUIREMENTS_FILE} ... "
  cp "${REQUIREMENTS_FILE}" "${REPO_NAME}"

  echo "Finding and Copying Python Files ... "
  find . -maxdepth 1 -type f -name "*${PYTHON_EXTENSION}" -exec cp {} "${REPO_NAME}" \;

}

__fix_requirements() {
  echo "Fixing ${REQUIREMENTS_FILE} ... "
  sed -i '/^sparseml/d' "${REPO_NAME}/${REQUIREMENTS_FILE}"

  yolact_file=".*yolact.py"
  echo "Fixing file level imports ... "
  find "${REPO_NAME}" -maxdepth 1 -type f -name "*${PYTHON_EXTENSION}" -print0 | while read -r -d $'\0' file; do
    echo "file is $file"
    if ! [[ "${file}" =~ $yolact_file ]]; then
      __fix_imports "${file}"
    fi
  done

  echo "Fixing folder level imports ... "
  for folder in "${NEEDED_FOLDERS[@]}"; do
    __fix_imports "${folder}"
  done

  echo "Fixing special case (yolact.py) imports ... "
  grep --include="*.py" -rnl "${REPO_NAME}/" -e '\(from\|import\) yolact ' | xargs -i@ sed -E -i "s/^(from|import) yolact /\1 ${REPO_NAME}.yolact /g" @

}

__fix_imports() {
  curr_import_name=$(echo "${1}" | sed -E "s/.*\/(.*)\.py$/\1/")

  echo "Fixing top level imports for ${curr_import_name} ... "
  grep --include="*.py" -rnl "${REPO_NAME}/" -e "import ${curr_import_name}" | xargs -i@ sed -i "s/^import ${curr_import_name} /import ${REPO_NAME}.${curr_import_name} /g" @

  echo "Fixing from imports for ${curr_import_name} ... "
  grep --include="*.py" -rnl "${REPO_NAME}/" -e "from ${curr_import_name}" | xargs -i@ sed -E -i "s/^from ${curr_import_name}([\\.| ])/from ${REPO_NAME}.${curr_import_name}\1/g" @

}

__build_setup() {
  # install build-tools
  python3 -m pip install --upgrade build

  package_name="$(python3 ${SETUP_FILE} --name)"
  package_version="$(python3 ${SETUP_FILE} --version)"

  if [[ -z "$package_name" || -z "$package_version" ]]; then
    echo "Could not determine package name/version (found ${package_name}==${package_version})" 1>&2
    return 4
  fi
  echo "Package: ${package_name}==${package_version}"

  # Cleanup old dist folder
  if [[ -e dist ]]; then
    echo "Removing old releases in dist/* ... "
    rm --recursive --force dist/ || return 5
  fi

  echo "NM_INTEGRATED=True" >>"${REPO_NAME}/__init__.py"
}

__test_build() {
  # shellcheck disable=SC2174
  mkdir --mode 700 --parents "${PACKAGE_TEST_DIR}" || return 11
  whl_file=$(find "dist/" -type f -name "*.whl")

  echo "Attempting to install ${whl_file}"
  python3 -m pip install --target "${PACKAGE_TEST_DIR}" --upgrade "${whl_file}"
  package_was_installed="$?"

  if [[ "${package_was_installed}" -ne "0" ]]; then
    echo "FAILED TO INSTALL ${whl_file} (status=${package_was_installed})" 1>&2
    return 13
  fi

  echo "Example change in imports .... for a file backbone.py"
  grep --include="*.py" -rn "${PACKAGE_TEST_DIR}/${REPO_NAME}/" -e "from yolact.backbone" || return 1

  echo "Example change in imports .... for a folder utils"
  grep --include="*.py" -rn "${PACKAGE_TEST_DIR}/${REPO_NAME}/" -e "from yolact.utils" || return 1
}

__teardown() {
  echo "Removing ${PACKAGE_TEST_DIR} ... "
  rm --recursive --force "${PACKAGE_TEST_DIR}" || return 12
}

main
