def calculate_barycenter_correction_kms(ra, dec, jd, site = 'keck'):
    '''
    If a precision of around 3 m/s is sufficient, the computed correction can then be added to any observed radial velocity to determine the final heliocentric radial velocity
    negative value means we were moving away from the source at the time of the observation, so lines should be redshifted. We apply a blueshift to correct for them. 
    '''
    from astropy.time import Time
    from astropy import units as u
    from astropy.coordinates import SkyCoord, EarthLocation
    
    loc = EarthLocation.of_site(site)

    sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    barycorr = sc.radial_velocity_correction(obstime= Time(jd, format = 'jd'), location=loc) 
    return barycorr.to(u.km/u.s).value


def heliocentric_correction(mjd_utc, ra, dec, site = 'Lick Observatory'):
    '''
    convert timestamps from mjd utc to HJD
    '''
    from astropy.time import Time
    from astropy import units as u
    from astropy.coordinates import SkyCoord, EarthLocation
    jd_utc = mjd_utc + 2400000.5
    
    loc = EarthLocation.of_site(site)
    time_inp = Time(jd_utc, format='jd', scale='utc', location=loc)

    target = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    ltt_helio = time_inp.light_travel_time(target, 'heliocentric')
    hjd_utc = jd_utc + ltt_helio
    
    return hjd_utc.value 
