#!/bin/bash
# Intentionally no `set -u`: TACC's Lmod `module` shell function references
# unset variables internally and would trip an unset-var trap.
set -eo pipefail

# ---- Constants ----

KERNEL_SRC=/scratch/projects/tacc/sci-training/cosmicai_2026/kernel_container/cosmicai26
KERNEL_NAME=cosmicai26
KERNEL_DST_ROOT=${HOME}/.local/share/jupyter/kernels
KERNEL_DST=${KERNEL_DST_ROOT}/${KERNEL_NAME}

MATERIALS_SRC=/scratch/projects/tacc/sci-training/sci_tacc_education_materials/cosmicai_26
MATERIALS_NAME=cosmicai_26
MATERIALS_DST_ROOT=${HOME}
MATERIALS_DST=${MATERIALS_DST_ROOT}/${MATERIALS_NAME}

# Extra content nested inside the materials folder; cleaned up automatically
# when MATERIALS_DST is removed during --uninstall.
EXAI_SRC=/scratch/projects/tacc/sci-training/TACC_exAI
EXAI_NAME=TACC_exAI
EXAI_DST_PARENT=${MATERIALS_DST}/Agentic_AI
EXAI_DST=${EXAI_DST_PARENT}/${EXAI_NAME}

MODULES="ucc/1.7.0 ucx/1.20.0 cmake/4.1.1 TACC gcc/15.1.0 openmpi/5.0.5 python3/3.11.8 sqlite/3.46.1 cuda/13.1 tacc-apptainer/1.4.1"

# Backup so --uninstall can restore the user's pre-install module default.
BACKUP_DIR=${HOME}/.cosmicai26_install
LMOD_DEFAULT=${HOME}/.lmod.d/default
LMOD_BACKUP=${BACKUP_DIR}/lmod_default
BACKUP_MARKER=${BACKUP_DIR}/had_prior_default

# ---- CLI defaults ----

ACTION=install
DRY_RUN=0
SAVE_MODULES=1
DO_MODULES=1
DO_KERNEL=1
DO_MATERIALS=1
STEP_SELECTED=0

usage() {
  cat <<'EOF'
Usage: install.sh [OPTIONS]

Sets up the CosmicAI 2026 tutorial environment: loads the required modules
(and saves them as the user's default), registers a Jupyter kernel, and
copies the course materials into $HOME.

Actions (default: --install):
  --install            Install everything (modules + kernel + materials)
  --uninstall          Remove what this script installed and restore the
                       module default that was in place before install
  --module-restore     Restore the pre-install module default only
                       (leaves the kernel and materials in place)
  -h, --help           Show this help

Step selection (default: all three; pass one or more to restrict):
  --modules-only       Only the module purge/load/save step
  --kernel-only        Only the Jupyter kernel step
  --materials-only     Only the course materials step

Modifiers:
  --no-module-save     Load the modules in this shell, but do not persist
                       them as the user default (install only)
  --dry-run            Print actions without executing them
EOF
}

# ---- Helpers ----

log_step() {
  echo
  echo "==> $*"
}

# run executes an external command, honoring --dry-run.
run() {
  echo "+ $*"
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

# run_shell evaluates a string in the current shell. Needed for `module ...`,
# which is a shell function rather than an external binary.
run_shell() {
  echo "+ $*"
  if [ "$DRY_RUN" -eq 0 ]; then
    eval "$*"
  fi
}

# ---- Argument parsing ----

select_step() {
  if [ "$STEP_SELECTED" -eq 0 ]; then
    DO_MODULES=0
    DO_KERNEL=0
    DO_MATERIALS=0
    STEP_SELECTED=1
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --install)       ACTION=install ;;
    --uninstall)     ACTION=uninstall ;;
    --module-restore) ACTION=uninstall; select_step; DO_MODULES=1 ;;
    --dry-run)       DRY_RUN=1 ;;
    --no-module-save) SAVE_MODULES=0 ;;
    --modules-only)   select_step; DO_MODULES=1 ;;
    --kernel-only)    select_step; DO_KERNEL=1 ;;
    --materials-only) select_step; DO_MATERIALS=1 ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# ---- Install steps ----

install_modules() {
  log_step "Configuring modules"
  run_shell "module purge"
  run_shell "module load $MODULES"

  if [ "$SAVE_MODULES" -eq 0 ]; then
    echo "(--no-module-save: not persisting modules as the default)"
    return
  fi

  run mkdir -p "$BACKUP_DIR"
  if [ -e "$LMOD_DEFAULT" ]; then
    if [ ! -e "$LMOD_BACKUP" ]; then
      run cp "$LMOD_DEFAULT" "$LMOD_BACKUP"
      run_shell "echo 1 > '$BACKUP_MARKER'"
    else
      echo "(existing backup at $LMOD_BACKUP preserved)"
    fi
  else
    if [ ! -e "$BACKUP_MARKER" ]; then
      run_shell "echo 0 > '$BACKUP_MARKER'"
    fi
  fi

  run_shell "module save"
  echo "Saved module set as default. To revert: $0 --module-restore"
}

install_kernel() {
  log_step "Installing Jupyter kernel ($KERNEL_NAME)"
  if [ -e "$KERNEL_DST" ]; then
    echo "Kernel already installed at $KERNEL_DST (skipping)."
    return
  fi
  run mkdir -p "$KERNEL_DST_ROOT"
  run cp -r "$KERNEL_SRC" "$KERNEL_DST_ROOT/"
}

install_materials() {
  log_step "Copying course materials ($MATERIALS_NAME)"
  if [ -e "$MATERIALS_DST" ]; then
    echo "Course materials already present at $MATERIALS_DST (skipping)."
  else
    run mkdir -p "$MATERIALS_DST_ROOT"
    run cp -r "$MATERIALS_SRC" "$MATERIALS_DST_ROOT/"
  fi

  if [ -e "$EXAI_DST" ]; then
    echo "$EXAI_NAME already present at $EXAI_DST (skipping)."
  else
    run mkdir -p "$EXAI_DST_PARENT"
    run cp -r "$EXAI_SRC" "$EXAI_DST_PARENT/"
  fi
}

# ---- Uninstall steps ----

uninstall_modules() {
  log_step "Restoring previous module default"
  if [ ! -e "$BACKUP_MARKER" ]; then
    echo "No backup marker at $BACKUP_MARKER; nothing to restore."
    return
  fi
  local had_prior
  had_prior=$(cat "$BACKUP_MARKER")
  if [ "$had_prior" = "1" ] && [ -e "$LMOD_BACKUP" ]; then
    run cp "$LMOD_BACKUP" "$LMOD_DEFAULT"
    echo "Restored previous module default from $LMOD_BACKUP."
  else
    if [ -e "$LMOD_DEFAULT" ]; then
      run rm "$LMOD_DEFAULT"
      echo "Removed module default (no prior default existed)."
    fi
  fi
  run rm -f "$BACKUP_MARKER" "$LMOD_BACKUP"
  rmdir "$BACKUP_DIR" 2>/dev/null || true
}

uninstall_kernel() {
  log_step "Removing Jupyter kernel ($KERNEL_NAME)"
  if [ -e "$KERNEL_DST" ]; then
    run rm -rf "$KERNEL_DST"
  else
    echo "Kernel not present at $KERNEL_DST (nothing to remove)."
  fi
}

uninstall_materials() {
  log_step "Removing course materials ($MATERIALS_NAME)"
  if [ ! -e "$MATERIALS_DST" ]; then
    echo "Materials not present at $MATERIALS_DST (nothing to remove)."
    return
  fi
  echo "About to remove $MATERIALS_DST and any local changes inside it."
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "(dry-run: would prompt for confirmation, then rm -rf)"
    return
  fi
  local reply=""
  read -r -p "Continue? [y/N] " reply || reply=""
  case "$reply" in
    y|Y|yes|YES) run rm -rf "$MATERIALS_DST" ;;
    *)           echo "Skipped materials removal." ;;
  esac
}

# ---- Dispatch ----

if [ "$DRY_RUN" -eq 1 ]; then
  echo "(dry-run mode: showing commands without executing them)"
fi

case "$ACTION" in
  install)
    if [ "$DO_MODULES"   -eq 1 ]; then install_modules;   fi
    if [ "$DO_KERNEL"    -eq 1 ]; then install_kernel;    fi
    if [ "$DO_MATERIALS" -eq 1 ]; then install_materials; fi
    ;;
  uninstall)
    if [ "$DO_MATERIALS" -eq 1 ]; then uninstall_materials; fi
    if [ "$DO_KERNEL"    -eq 1 ]; then uninstall_kernel;    fi
    if [ "$DO_MODULES"   -eq 1 ]; then uninstall_modules;   fi
    ;;
esac
