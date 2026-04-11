from typing import Any

__all__ = [
    "CellDescriptor",
    "MBTRDescriptor",
    "OVFDescriptors",
    "SOAPDescriptor",
]


def __getattr__(name: str) -> Any:
    if name == "CellDescriptor":
        from descriptors.cell import CellDescriptor
        return CellDescriptor
    if name == "MBTRDescriptor":
        from descriptors.mbtr import MBTRDescriptor
        return MBTRDescriptor
    if name == "OVFDescriptors":
        from descriptors.ovf_by_dscribe import OVFDescriptors
        return OVFDescriptors
    if name == "SOAPDescriptor":
        from descriptors.soap import SOAPDescriptor
        return SOAPDescriptor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
