from owcurate.Files.Converters import *


def bin_to_edf(file_in, out_path, accel=True, temperature=True, light=False, button=False):
    """Function that calls Files.Converters from owcurate to convert .bin file to EDF format.
       Each sensor (i.e. accelerometer, light, temperature, button) are saved as individual files.

    :argument
    -file_in: string, pathway for .bin file (including ".bin")
    -out_path: where file(s) is/are saved. NOTE: location requires folders titled "Accelerometer", "Light",
               "Temperature", and "Button"
    -accel, temperature, light, button: boolean of whether to save this sensor to file.
                                        Only accelerometer is True by default.

    :returns
    -GENEActiv class instance
    """

    # Converts boolean arguments to correct format for GENEActivToEDF call
    if accel:
        save_accel = "Accelerometer"
    if not accel:
        save_accel = ""
    if temperature:
        save_temp = "Temperature"
    if not temperature:
        save_temp = ""
    if light:
        save_light = "Light"
    if not light:
        save_light = ""
    if button:
        save_button = "Button"
    if not button:
        save_button = ""

    output_dict = {"Accelerometer": accel, "Temperature": temperature, "Light": light, "Button": button}

    print("\n" + "Converting and saving files:")
    print(output_dict)
    print("\n")

    geneactiv_object = GENEActiv()  # Creates GENEActiv class instance
    geneactiv_object.read_from_raw(path=file_in)  # Reads .bin file
    geneactiv_object.calculate_time_shift()  # Performs time shift (clock drift)

    # Conversion
    GENEActivToEDF(GENEActiv=geneactiv_object, path=out_path,
                   accel=save_accel, temperature=save_temp,
                   light=save_light, button=save_button)

    return geneactiv_object
