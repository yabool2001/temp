import time as t

def milis_timestamp () -> str :
    return f"{int ( t.time () * 1000 )}"