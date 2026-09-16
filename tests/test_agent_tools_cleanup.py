"""Exercise the actual smoke controller's cleanup without booting a VM."""
import ast
from pathlib import Path
import subprocess

import pytest


def cleanup(controller):
    path = Path(__file__).parents[1]/'deploy/public/worker'/controller
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_cleanup')
    namespace = {'subprocess':subprocess}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['_cleanup']


@pytest.mark.parametrize('controller', ['agent-tools-backend-smoke.py', 'agent-mcp-backend-smoke.py', 'agent-pi-mcp-backend-smoke.py'])
@pytest.mark.parametrize('failure', ['timeout', 'terminate', 'kill'])
def test_relay_failure_never_skips_vm_stop(failure, controller):
    stopped, killed, waits = [], [], []
    class Backend:
        def stop(self, handle):
            stopped.append(handle)
        def inventory(self):
            return set()
    class Relay:
        def terminate(self):
            if failure == 'terminate':
                raise OSError('synthetic terminate failure')
        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired('synthetic relay', timeout)
        def kill(self):
            killed.append(True)
            if failure == 'kill':
                raise OSError('synthetic kill failure')
    operation = cleanup(controller)
    if failure == 'timeout':
        operation(Backend(),'job',Relay())
        assert killed == [True] and len(waits) == 2
    else:
        with pytest.raises(OSError):
            operation(Backend(),'job',Relay())
    assert stopped == ['job']
