#!/usr/bin/python

"""
A simple discretionary locking system for /dev/nvidia devices.

Iain Murray, November 2009, January 2010, January 2011.
"""

import os, os.path

_dev_prefix = '/dev/nvidia'
#URL = 'http://www.cs.toronto.edu/~murray/code/gpu_monitoring/'
URL = 'http://homepages.inf.ed.ac.uk/imurray2/code/gpu_monitoring/'


# Get ID's of NVIDIA boards. Should do this through a CUDA call, but this is
# a quick and dirty way that works for now:
def board_ids():
    """Returns integer board ids available on this machine."""
    from glob import glob
    board_devs = glob(_dev_prefix + '[0-9]*')
    return range(len(board_devs))

def _lock_file(id):
    """lock file from integer id"""
    # /tmp is cleared on reboot on many systems, but it doesn't have to be
    if os.path.exists('/dev/shm'):
        # /dev/shm on linux machines is a RAM disk, so is definitely cleared
        return '/dev/shm/gpu_lock_%d' % id
    else:
        return '/tmp/gpu_lock_%d' % id

def owner_of_lock(id):
    """Username that has locked the device id. (Empty string if no lock)."""
    import pwd
    try:
        statinfo = os.lstat(_lock_file(id))
        return pwd.getpwuid(statinfo.st_uid).pw_name
    except:
        return ""

def _obtain_lock(id):
    """Attempts to lock id, returning success as True/False."""
#    print   id
    try:
        # On POSIX systems symlink creation is atomic, so this should be a
        # robust locking operation:
        os.symlink('/dev/null', _lock_file(id))
        return True
    except:
        return False

def _launch_reaper(id, pid):
    """Start a process that will free a lock when process pid terminates"""
    from subprocess import Popen, PIPE
    me = __file__
    if me.endswith('.pyc'):
        me = me[:-1]
    myloc = os.path.dirname(me)
    if not myloc:
        myloc = os.getcwd()
    reaper_cmd = os.path.join(myloc, 'run_on_me_or_pid_quit')
    Popen([reaper_cmd, str(pid), me, '--free', str(id)],
        stdout=open('/dev/null', 'w'))

def obtain_lock_id(pid = None):
    """
    Finds a free id, locks it and returns integer id, or -1 if none free.

    A process is spawned that will free the lock automatically when the
    process pid (by default the current python process) terminates.
    """
    id = -1
    id = obtain_lock_id_to_hog()
    try:
        if id >= 0:
            if pid is None:
                pid = os.getpid()
            _launch_reaper(id, pid)
    except:
        free_lock(id)
        id = -1
    return id

def obtain_lock_id_to_hog():
    """
    Finds a free id, locks it and returns integer id, or -1 if none free.

    * Lock must be freed manually *
    """
    for id in board_ids():
        if _obtain_lock(id):
            return id
    return -1

def free_lock(id):
    """Attempts to free lock id, returning success as True/False."""
    try:
        filename = _lock_file(id)
        # On POSIX systems os.rename is an atomic operation, so this is the safe
        # way to delete a lock:
        os.rename(filename, filename + '.redundant')
        os.remove(filename + '.redundant')
        return True
    except:
        return False


# If run as a program:
if __name__ == "__main__":
    import sys
    me = sys.argv[0]
    # Report
    if '--id' in sys.argv:
        if len(sys.argv) > 2:
            try:
                pid = int(sys.argv[2])
                print   pid, sys.argv[2]
                assert(os.path.exists('/proc/%d' % pid))
            except:
                print 'Usage: %s --id [pid_to_wait_on]' % me
                print 'The optional process id must exist if specified.'
                print 'Otherwise the id of the parent process is used.'
                sys.exit(1)
        else:
            pid = os.getppid()
            print   pid
        print obtain_lock_id(pid)
    elif '--id-to-hog' in sys.argv:
        print obtain_lock_id_to_hog()
    elif '--free' in sys.argv:
        try:
            id = int(sys.argv[2])
        except:
            print 'Usage: %s --free <id>' % me
            sys.exit(1)
        if free_lock(id):
            print "Lock freed"
        else:
            owner = owner_of_lock(id)
            if owner:
                print "Failed to free lock id=%d owned by %s" % (id, owner)
            else:
                print "Failed to free lock, but it wasn't actually set?"
    else:
        print '\n  Usage instructions:\n'
        print '  To obtain and lock an id: %s --id' % me
        print '  The lock is automatically freed when the parent terminates'
        print
        print "  To get an id that won't be freed: %s --id-to-hog" % me
        print "  You *must* manually free these ids: %s --free <id>\n" % me
        print '  More info: %s\n' % URL
        div = '  ' + "-"*60
        print '\n' + div
        print "  NVIDIA board users:"
        print div
        for id in board_ids():
            print "      Board %d: %s" % (id, owner_of_lock(id))
        print div + '\n'
