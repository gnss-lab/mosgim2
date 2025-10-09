from numpy import sin, cos, pi, arcsin

RE = 6371.2
HM=300

def sub_ionospheric(s_lat, s_lon, hm, az, el, R=RE):
    """
    Calculates subionospheric point and delatas from site
    Parameters:
        s_lat, slon - site latitude and longitude in radians
        hm - ionposheric maximum height (km)
        az, el - azimuth and elevation of the site-sattelite line of sight in
            radians
        R - Earth radius (km)
    """
    #TODO use meters
    psi = pi / 2 - el - arcsin(cos(el) * R / (R + hm))
    lat = bi = arcsin(sin(s_lat) * cos(psi) + cos(s_lat) * sin(psi) * cos(az))
    lon = sli = s_lon + arcsin(sin(psi) * sin(az) / cos(bi))
    
    lon[lon > pi] = lon[lon > pi] - 2 * pi 
    lon[lon < -pi] = lon[lon < -pi] + 2 * pi 
    return lat, lon