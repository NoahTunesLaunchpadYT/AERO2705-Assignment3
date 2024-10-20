import math

# Function to calculate the Julian day at 0 hr UT for any date
def J0(year, month, day):
    """
    Computes the Julian day number at 0 UT (Universal Time) for any year between 1900 and 2100.
    
    Args:
        year (int): Year between 1901 and 2099.
        month (int): Month (1 - 12).
        day (int): Day of the month (1 - 31).
        
    Returns:
        float: Julian day at 0 UT.
    """
    j0 = 367 * year - math.floor(7 * (year + math.floor((month + 9) / 12)) / 4) \
         + math.floor(275 * month / 9) + day + 1721013.5
    return j0

# Function to reduce an angle to the range 0 - 360 degrees
def zeroTo360(angle):
    """
    Reduces an angle to the range 0 - 360 degrees.
    
    Args:
        angle (float): Angle in degrees.
        
    Returns:
        float: Reduced angle in the range [0, 360).
    """
    angle = angle % 360
    if angle < 0:
        angle += 360
    return angle

# Function to convert a day of the year with fractional part to day and UT in hours
def day_of_year_to_ut(day_of_year):
    """
    Converts the fractional day of the year to Universal Time (UT).
    
    Args:
        day_of_year (float): The day of the year with fractional part.
    
    Returns:
        tuple: A tuple containing (day, ut) where day is the integer day of the year and ut is the Universal Time in hours.
    """
    day = int(day_of_year)  # Integer part is the day of the year
    ut = (day_of_year - day) * 24  # Fractional part converted to hours
    return day, ut

# Function to extract epoch year and day from TLE line 1
def extract_epoch_from_tle(line1):
    """
    Extracts the epoch (year and day) from line 1 of the TLE.
    
    Args:
        line1 (str): Line 1 of the TLE.
    
    Returns:
        tuple: A tuple containing (year, day_of_year).
    """
    epoch_year = int(line1[18:20].strip())  # Extract 2-digit year
    epoch_day = float(line1[20:32].strip())  # Extract day of year with fractional part
    
    # Convert 2-digit year to 4-digit year
    if epoch_year < 57:
        epoch_year += 2000
    else:
        epoch_year += 1900

    return epoch_year, epoch_day

# Function to calculate Local Sidereal Time (LST), which is GST at Greenwich
def LST(year, month, day, ut, EL):
    """
    Calculates the Local Sidereal Time (LST) or GST if longitude is set to 0.
    
    Args:
        year (int): Year.
        month (int): Month.
        day (int): Day.
        ut (float): Universal Time (hours).
        EL (float): East Longitude (degrees).
        
    Returns:
        float: Local Sidereal Time in degrees.
    """
    # Calculate Julian Day Number at 0 hr UT
    j0 = J0(year, month, day)
    
    # Calculate the number of centuries since J2000
    j = (j0 - 2451545.0) / 36525
    
    # Calculate Greenwich Sidereal Time at 0 hr UT (Equation 5.50)
    g0 = 100.4606184 + 36000.77004 * j + 0.000387933 * j**2 - 2.583e-8 * j**3
    
    # Reduce g0 to the range 0 - 360 degrees
    g0 = zeroTo360(g0)
    
    # Calculate Greenwich Sidereal Time at the specified UT (Equation 5.51)
    gst = g0 + 360.98564724 * (ut / 24)
    
    # Calculate Local Sidereal Time (Equation 5.52)
    lst = gst + EL
    
    # Reduce lst to the range 0 - 360 degrees
    lst = zeroTo360(lst)
    
    return lst

# Function to calculate GST from Line 1 of TLE
def calculate_gst_from_tle(line1):
    """
    Calculates the Greenwich Sidereal Time (GST) at the epoch specified in the TLE.
    
    Args:
        line1 (str): Line 1 of the TLE.
    
    Returns:
        float: Greenwich Sidereal Time (GST) in degrees.
    """
    # Extract the epoch from the TLE
    epoch_year, epoch_day = extract_epoch_from_tle(line1)
    
    # Convert the epoch day to the day of year and UT
    day, ut = day_of_year_to_ut(epoch_day)
    
    # Choose a fixed longitude for GST (Greenwich = 0 degrees)
    longitude_greenwich = 0
    
    # Calculate the Greenwich Sidereal Time (GST) using the LST function
    gst = LST(epoch_year, 1, day, ut, longitude_greenwich)
    
    return gst

# Example usage
if __name__ == "__main__":
    # Example TLE Line 1
    line1 = "1 70360C 24149BM  24235.92391204  .00016145  00000+0  79964-3 0  2359 \n"
    
    # Calculate GST (starting meridian angle in vernal equinox)
    gst = calculate_gst_from_tle(line1)
    print(f"Starting Meridian Sidereal Angle (GST) in Vernal Equinox: {gst:.6f} degrees")
