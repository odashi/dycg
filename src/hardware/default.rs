use once_cell::sync::OnceCell;
use std::sync::Mutex;

use crate::hardware::cpu::Cpu;
use crate::hardware::Hardware;

/// Returns the default hardware.
///
/// Any arrays without explicit specification of hardware falls back to use this hardware.
pub(crate) fn get_default_hardware() -> &'static Mutex<Box<dyn Hardware>> {
    static SINGLETON: OnceCell<Mutex<Box<dyn Hardware>>> = OnceCell::new();
    SINGLETON.get_or_init(|| Mutex::new(Box::new(Cpu::new("default"))))
}