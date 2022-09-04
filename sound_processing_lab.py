import numpy as np
from scipy.io import wavfile

"""
Sound Processing Lab - A simple DSP-application where the user can:
1) Upload a sound file
2) Apply a chorus, delay or distortion filter according to customizable settings.
3) Download a new sound file of the filtered audio.

Version last edited: 4 september 2022.
Contact: david.larsson-holmgren@hotmail.com
"""

class Sound_file():
    """
    A class for sound file objects, storing data and meta data of uploaded sound files.
    """
    def __init__(self, data, sf):
        """
        A sound file-object constructor.
        :param data: The sound data of the sound file.
        :param sf: The sample frequency of the sound file.
        """
        self.data = data
        self.sf = sf
        self.filter = None
        self.filter_params = dict()
        self.file_name = None

    def __str__(self):
        """
        Displaying meta data attributes of the sound file object.
        :return: Return a printed string of the size of the data array and the sample frequency.
        """
        return "Pre-filtered file name: {}\
            \nData size: {}\
            \nSample frequency: {}\
            \nFilter applied: {}\
            \nFilter parameters: {}"\
            .format(self.file_name, self.data.size, self.sf, self.filter, self.filter_params)


"""
        ~~~~~~~~~~~~~~~~~~~ THE FILTER FUNCTIONS ~~~~~~~~~~~~~~~~~~~
"""

def chorus(X):
    """
    A chorus filter with the delayed signal variating according to a sine wave.
    :param X: A sound object
    :return filt_obj: Returning the sound object of the output signal.
    """
    # The filter parameters, available for custom setting.
    alpha = 0.25 # The amplification of the input signal, preferred interval [0, 1]
    beta = 0.25 # The amplification of delayed signal, preferred interval [0, 1]
    fi = 0.1 # Modulation depth, preferred interval [0, 1]
    f = 0.25 # The modulation frequency. For flanger effect.
    t = 50 # Average time delay in ms. For more of an flanger effect, try 10 ms.

    # Saving filter parameters in dict.
    settings = {"alpha": alpha, "beta": beta, "fi": fi, "f": f, "t": t}

    D = int((t*10**(-3))*X.sf) # Calculating the average time delay t in samples.
    Y = np.zeros(X.data.shape) # Creating the output signal array.
    X_norm = normalize(X.data) # Normalizing the input signal.

    # Processing the normalized input signal X_norm
    for n in range(X_norm.shape[0]):
        # Calculating the index of the delayed signal according to a sine wave variation.
        delayed_index = int(D*(1+fi*np.sin(2*np.pi*f/X.sf*n))) 
        if n - delayed_index < 0: # Avoiding negative indices.
            Y[n] = alpha*X_norm[n]
        else: # Adding the chorus signal depending on the variating delayed_index.
            Y[n] = alpha*X_norm[n] + beta*X_norm[n - delayed_index]

    # creating new filtered sound object, same sf as original.
    filt_obj = create_sound_file_object(Y, X.sf)
    filt_obj.filter = "chorus" # saving which effect was added
    filt_obj.filter_params = settings # saving the settings of the filter parameters.

    return filt_obj


def delay(X):
    """
    A delay filter with added feedback.
    :param X: A sound object
    :return filt_obj: Returning the sound object of the output signal.
    """
    
    # The filter parameters, available for custom setting.
    alpha = 0.4 # The amplification of the delayed input signal, preferred interval [0, 1]
    beta = 0.15 # The amplification of the feedback signal, preferred interval [0, 1]
    t = 430 # Average time delay in ms.

    # Saving filter parameters in dict.
    settings = {"alpha": alpha, "beta": beta, "t": t} 

    D = int((t*10**(-3))*X.sf) # Calculating the average time delay t in samples.
    D2 = int(D*2) # Calculating the average time delay for the feedback signal.
    Y = np.zeros(X.data.size) # Creating the output signal array.

    X_norm = normalize(X.data) # Normalizing the input signal.

    # Processing the input signal
    for n in range(X_norm.shape[0]):
        if n-D2 < 0: # Avioding negative indicies.
            Y[n] = X_norm[n]
        else: # Adding the delays to the output signal.
            Y[n] = X_norm[n] + alpha*X_norm[n-D] + beta*Y[n-D2]

    # creating new filtered sound object, same sf as original.
    filt_obj = create_sound_file_object(Y, X.sf)
    filt_obj.filter = "delay" # saving which effect was added
    filt_obj.filter_params = settings # saving the parameters settings made.

    return filt_obj

def distortion(X):
    """
    A cubic distortion filter.
    :param X: A sound object
    :return filt_obj: Returning the sound object of the output signal.
    """

    # The filter parameter, available for custom setting.  
    dist_coef = 0.2 # Distortion coefficient for the distortion, interval (0, 1]. 
    # Values closer to 0 gives more distortion, vice versa.

    # Saving filter parameters in dict.
    settings = {"dist_coef": dist_coef} 

    X_norm = normalize(X.data) # Normalizing the input signal.
    Y = np.zeros(X.data.shape) # Creating the output signal array.

    # Processing the output signal
    for n in range(X_norm.shape[0]):
        # Clipping all values above the distortion coefficient.
        if X_norm[n] < - dist_coef:
            Y[n] = - dist_coef
        # Clipping all values below the distortion coefficient.
        elif X_norm[n] > dist_coef:
            Y[n] = dist_coef
        else:
            # Passing the other values according to the original input signal.
            Y[n] = X_norm[n]

    # creating new filtered sound object, same sf as original.
    filt_obj = create_sound_file_object(Y, X.sf)
    filt_obj.filter = "distortion" # saving which effect was added
    filt_obj.filter_params = settings # saving the parameters settings made.
    
    return filt_obj


"""
        ~~~~~~~~~~~~~~~~~~~ OTHER FUNCTIONS ~~~~~~~~~~~~~~~~~~~
"""

def get_file_name():
    """
    A function letting the user enter the file name to be processed.
    Error handling is implemented to handle wrong file name and format.
    :return:  Sampling frequency (sf), the data of the sound file, file name. 
    """
    # Message for input
    message = "\nPlease enter the name of the wav-file you want to process:\n"
    
    # Description of accepted input of the user.
    help_message = "\nImportant:\n* Don't use parentheses.\
        \n* Only wav-format is allowed.\
        \n* If your wav-file is located in another folder, don't forget to add the pathname.\
    (e.g. /Users/username/Music/sound.wav)\n"
    
    # Messages if error should occur.
    error_message_1 = "\nThe file name you have entered does not exist. Please try again.\n"
    # An enter name input.
    error_message_2 = "\nWrong file format. Only WAV-files are accepted.\
        Please try another file.\n"
    enter_name = "\nPlease enter file name: "

    # boolean value controlling the while loop of the error handling
    valid=False
    while not valid:
        # Displays instructions for the user to input file name.
        file_name = input(message+help_message+enter_name)
        # Try to to read sound file.
        try:
            sf, sound_data = wavfile.read(file_name)
        # if file not found, new specifed instructions are given.
        # letting the user try again
        except FileNotFoundError:
            message = error_message_1
        # If wrong format is given by user, new specifed instructions are given.
        # letting the user try again
        except ValueError:
            message = error_message_2
        else:
            # When correct file name and is provided by the user
            # the sample frequency, sound data and file name is returned
            return sf, sound_data, file_name


def get_integer():
    """
    A function taking input from user, asking a integer in the range of [1, 3].
    Error handling is implemented for avoiding incorrect input.
    :return: The integer chosen by the user in the range of [1, 3].
    """
    # descriptive messages given to the user
    message = "\nPlease choose one of the following sound effects (1, 2 or 3):"
    alternatives = "\n1 Chorus\n2 Delay\n3 Distortion\n"
    your_choice = "\nYour choice: "
    error_message = "\nIncorrect choice.\nPlease choose 1, 2 or 3:"
    
    # boolean value controlling the while loop of the error handling
    valid=False
    while not valid:
        # the user providing input based on given instructions
        int_choice = input(message+alternatives+your_choice)
        try:
            # testing if given input is a integer
            int_choice = int(int_choice)
        # if value error, changing the input message to error message
        except ValueError:
            message = error_message
        else:
            # if integer is given but outside the range of [1, 3]
            valid = 0 < int_choice < 4
            if not valid:
                # changing message to input message to error message
                message = error_message
            else:
                # return a correct given user in the range of [1, 3]
                return int_choice


def create_sound_file_object(sound_data, sf):
    """
    Returning a sound file object.
    :param sound_data: The data of the sound file.
    :param sf: The sample frequency of the sound file.
    :return: Returning the object.
    """
    return Sound_file(sound_data, sf)


def create_new_file_name(sound_obj):
    """
    Creating a new file name for the filtered sound, based on the previous file name,
    the filter applied and the settings of the filter.
    :param sound_obj: A sound object.
    :return: A new file name for the filtered sound file.
    """
    # Adding the previous file name to the new file name string. Removing ".wav"
    # adding "_" for readibility.
    file_name = sound_obj.file_name[:len(sound_obj.file_name)-4]+"_"
    # Adding a "("-symbol for the filter settings.
    file_name += sound_obj.filter+"("
    # A for-loop adding the settings of the filter to the new file name,
    # using the filter parameter-attribute of the particular sound object.
    for i in sound_obj.filter_params.keys():
        file_name += i+"_"+str(sound_obj.filter_params[i])+"_"
    # Ending the the filname with end ")"-symbol and the .wav format.
    file_name+=").wav"

    return file_name


def write_new_sound_file(sound_obj):
    """
    Creating a file name and writing a new sound file for the new filtered sound.
    :param sound_file: A sound file object.
    """
    # Noramlizing the sound data to avoid clipping.
    norm_sound = normalize(sound_obj.data)
    print("Creating new file name...")
    # Creating a new file name based on the filter and the fitler parameters (settings)
    file_name = create_new_file_name(sound_obj)
    print("New file complete.")
    print("Creating new sound file...")
    # Writing the new sound fil, wav-format.
    wavfile.write(file_name, sound_obj.sf, norm_sound)
    # Feedback for the user that the sound file is created and its name.
    print('\nYour processed sound file "' + file_name +' is now created.')


def normalize(X):
    """
    Normalizes input array x according to the value of x with the highest amplitude.
    :param X: Input numpy array.
    :return: Returning a new normalized version of X.
    """
    return X/np.max(np.abs(X))


def sound_processing(int_choice, sound_obj):
    """
    A function processing the sound by calling the different filter functions based
    on user integer input.
    :param int_choice: The integer choice of [1, 3] received by the user.
    :param sound_obj: A sound object.
    :return: Returning the sound object of the filtered sound.
    """
     # A variable with placeholder value, prepared for the filtered sound oubject
    filt_obj = None # TEST!!
    print("\nProcessing sound...")
    # Calling the different filter functions depending on the choice of the user.
    if int_choice == 1:
        filt_obj = chorus(sound_obj)
    if int_choice == 2:
        filt_obj = delay(sound_obj)
    if int_choice == 3:
        filt_obj = distortion(sound_obj)
    print("Processing completed.")
    return filt_obj # Returning the sound object of the filtered sound.


def welcome_message():
    """
    Printing simple welcome string.
    :return: Returning a welcome message.
    """
    return print("\nWelcome to Sound Processing Lab!")


def end_message():
    """
    A final message before exiting. Instructing the user to restart the program to
    process more files.
    :return: A string of instructions.
    """
    return print("\nThank you for using Sound Processing Lab. \
        \nTo process another file, please restart.\
    \nIf not, have a great day!\n")
  

def main():
    """
    The main function from which the program is running.
    """
    welcome_message()
    # Receiving the sample frequency, sound data and file name
    sf, sound_data, file_name = get_file_name()
    # creating the sound object for storing meta data of the sound.
    sound_obj = create_sound_file_object(sound_data, sf)
    # Getting input form user which filter is to be applies to the sound.
    int_choice = get_integer()
    # processing the sound based on user input of filter.
    filt_obj = sound_processing(int_choice, sound_obj)
    # saving previous file name to new filtered soudn object
    filt_obj.file_name = file_name
    # writing new sound file
    write_new_sound_file(filt_obj)
    end_message()

main()
