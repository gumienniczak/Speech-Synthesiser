"""A basic speech synthesiser"""
import string
import wave
import re
import os
import numpy as np
from nltk.corpus import cmudict
import simpleaudio
from synth_args import process_commandline


class Synth:
    """Class which allows loading diphone inventory and synthesising
    a list of phones using these diphones"""

    def __init__(self, wav_folder):
        self.wav_folder = wav_folder
        self.rate = simpleaudio.RATE
        self.diphones = self.get_wavs(wav_folder)

    def get_wavs(self, wav_folder):
        """Loads all the waveform data contained in WAV_FOLDER.
        Returns a dictionary, with unit names as keys and the corresponding
        loaded audio data as values."""

        diphones = {}

        audio_instance = simpleaudio.Audio()
        for file in os.listdir(wav_folder):
            # only corrupt .wav files terminate the program
            if 'wav' in file:
                try:
                    file_path = os.path.join(wav_folder, file)
                    audio_instance.load(file_path)
                    diphones[str(file)] = audio_instance.data
                # terminate program if corrupted .wav file is found
                except wave.Error:
                    raise
            else:
                pass

        self.rate = audio_instance.rate
        return diphones

    def smooth_end(self, diphone_array, nr_of_values):
        """
        Smooths the values at the end of an array for a given nr of values.
        :param diphone_array: an array
        :param nr_of_values: desired number of values
        :return array with smoothed out end
        """
        working_range = range(nr_of_values)
        list_of_factors = [i / nr_of_values for i in working_range]
        values_and_factors = zip(diphone_array[-nr_of_values:], reversed(list_of_factors))
        values_updated = [int(a * b) for a, b in values_and_factors]
        for i in working_range:
            diphone_array[-nr_of_values:][i] = values_updated[i]
        return diphone_array

    def smooth_beg(self, diphone_array, nr_of_values):
        """
        Smooths the values at the beginning of an array for a given nr of values.
        :param diphone_array: an array
        :param nr_of_values: desired number of values
        :return: array with smoothed out beginning
        """
        list_of_factors = [i / nr_of_values for i in range(nr_of_values)]
        values_and_factors = zip(diphone_array[:nr_of_values], list_of_factors)
        values_updated = [int(a * b) for a, b in values_and_factors]
        for i in range(nr_of_values):
            diphone_array[:nr_of_values][i] = values_updated[i]
        return diphone_array

    def synthesise(self, phones, reverse=False, smooth_concat=False):
        """
        Synthesises a phone list into an audio utterance.
        :param reverse: Whether to reverse the output signal. Either "signals" or None.
        :param phones: list of phones (list of strings)
        :param smooth_concat:
        :return: synthesised utterance (Audio instance)
        """

        audio_instance = simpleaudio.Audio(rate=self.rate)
        concat_ary = np.array([], dtype=audio_instance.nptype)
        list_for_smoothing = []
        affected_values = int(self.rate / 100)
        for diphone in self.phones_to_diphones(phones):
            try:
                if smooth_concat:
                    list_for_smoothing.append(self.diphones[diphone])
                else:
                    concat_ary = np.concatenate((concat_ary, self.diphones[diphone]))
            except KeyError:
                print(f'Missing diphone: {diphone}, proceeding as normal.')
                pass

        if smooth_concat:
            # first smooth the diphones
            smooth_dip = [self.smooth_end(list_for_smoothing[0], affected_values)]
            for element in list_for_smoothing[1:-1]:
                self.smooth_beg(element, affected_values)
                self.smooth_end(element, affected_values)
                smooth_dip.append(element)
            smooth_dip.append(self.smooth_beg(list_for_smoothing[-1], affected_values))

            # next extract the values for overlap and add them together
            val_to_add = []
            for i in range(len(smooth_dip)):
                try:
                    val_to_add.append(np.add(smooth_dip[i][-affected_values:],
                                             smooth_dip[i + 1][:affected_values]))
                except IndexError:
                    break

            # split diphones into parts which need trimming
            dip_val_border = [np.split(smooth_dip[0], [-affected_values])]
            dip_val_middle = []
            for element in smooth_dip[1:-1]:
                dip_val_middle.append(np.split(element, [affected_values, -affected_values]))
            dip_val_border.append(np.split(smooth_dip[-1], [affected_values]))

            # extract the correct arrays from the lists created above
            dip_arr = [dip_val_border[0][0]]
            for element in dip_val_middle:
                dip_arr.append(element[1])
            dip_arr.append(dip_val_border[1][1])

            concat_ary = np.concatenate((concat_ary, dip_arr[0]))
            for i in range(len(val_to_add)):
                concat_ary = np.concatenate((concat_ary, val_to_add[i]))
                concat_ary = np.concatenate((concat_ary, dip_arr[i + 1]))

        audio_instance.data = concat_ary

        if reverse == 'signal':
            audio_instance.data = np.flip(audio_instance.data)
        return audio_instance

    def phones_to_diphones(self, phones):
        """
        Converts a list of phones to the corresponding diphone units
        (to match units in diphones folder).
        :param phones: list of phones (list of strings)
        :return: list of diphones (list of strings)
        """
        l_phones = []
        l_phones_fil = []
        l_diphones = []
        for phone in phones:
            l_phones.append(phone.lower())

        # deletes stress information
        for phone in l_phones:
            numbers = re.compile(r'[0-9]')
            l_phones_fil.append(re.sub(numbers, '', phone))

        l_phones_fil.insert(0, 'pau')
        l_phones_fil.append('pau')

        for i in range(len(l_phones_fil)):
            try:
                l_diphones.append(l_phones_fil[i] + '-' + l_phones_fil[i + 1] + '.wav')
            except IndexError:
                break

        return l_diphones


class Utterance:
    """Class which supports phrase normalisation,
    a phrase can be turned into list of strings"""

    def __init__(self, phrase):
        """
        Constructor takes a phrase to process.
        :param phrase: a string which contains the phrase to process.
        """
        self.phrase = phrase

    date_format = re.compile(r'(\d+)/(\d+)/?(\d{2,})?')

    def normalise_phrase(self):
        """Strips a phrase from punctuation, makes everything lowercase"""
        normalised_words = []
        date_lookup = self.date_format.search(self.phrase)
        if date_lookup:
            date_group = date_lookup.group(0)
            self.phrase = self.phrase.replace(date_group, self.normalise_dates(date_group))
            # employing a recursive function until all dates are converted accordingly
            self.normalise_phrase()
        for word in self.phrase.split():
            word_without_punct = word.translate(str.maketrans('', '', string.punctuation))
            normalised_words.append(word_without_punct.lower())
        return normalised_words

    def normalise_dates(self, date):
        """Normalises the dates by classifying them into categories
         and looking up appropriate dictionary values"""
        dys = {'1': 'first', '2': 'second', '3': 'third', '4': 'fourth', '5': 'fifth',
               '6': 'sixth', '7': 'seventh', '8': 'eighth', '9': 'ninth', '10': 'tenth',
               '11': 'eleventh', '12': 'twelfth', '13': 'thirteenth', '14': 'fourteenth',
               '15': 'fifteenth', '16': 'sixteenth', '17': 'seventeenth', '18': 'eighteenth',
               '19': 'nineteenth', '20': 'twentieth', '21': 'twenty-first', '22': 'twenty-second',
               '23': 'twenty-third', '24': 'twenty-fourth', '25': 'twenty-fifth',
               '26': 'twenty-sixth', '27': 'twenty-seventh', '28': 'twenty-eighth',
               '29': 'twenty-ninth', '30': 'thirtieth', '31': 'thirty-first'}
        mths = {'1': 'january', '2': 'february', '3': 'march', '4': 'april',
                '5': 'may', '6': 'june', '7': 'july', '8': 'august', '9': 'september',
                '10': 'october', '11': 'november', '12': 'december'}
        yrs_ones = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six',
                    '7': 'seven', '8': 'eight', '9': 'nine'}
        yrs_tens = {'0': 'o', '2': 'twenty', '3': 'thirty', '4': 'forty', '5': 'fifty',
                    '6': 'sixty', '7': 'seventy', '8': 'eighty', '9': 'ninety'}
        yrs_10to19 = {'10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
                      '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
                      '18': 'eighteen', '19': 'nineteen'}

        matched_date = re.match(self.date_format, date)
        day = matched_date.group(1).lstrip('0')
        month = matched_date.group(2).lstrip('0')
        year = matched_date.group(3)

        if year:
            if len(year) != 2:
                year = year[-2:]
            if year in yrs_10to19:
                d_str = f'{mths[month]} {dys[day]} nineteen {yrs_10to19[year]}'
            else:
                d_str = f'{mths[month]} {dys[day]} nineteen {yrs_tens[year[0]]} {yrs_ones[year[1]]}'
        else:
            d_str = f'{mths[month]} {dys[day]}'

        return d_str

    def get_phone_seq(self, reverse=None):
        """
        Returns the phone sequence corresponding to the text in this Utterance (i.e. self.phrase)
        :param reverse:  Whether to reverse something.  Either "words", "phones" or None
        :return: list of phones (as strings)
        """
        dictionary = cmudict.dict()
        phrase = self.normalise_phrase()

        if reverse == 'words':
            phrase.reverse()

        list_of_phones = []
        for word in phrase:
            # look up each word in CMU dictionary, then retrieve first entry for word
            try:
                for phone in dictionary[word][0]:
                    list_of_phones.append(phone)
            except KeyError:
                print(f'Word not found in the dictionary: {word}')

        if reverse == 'phones':
            list_of_phones.reverse()

        return list_of_phones


def process_file(textfile, args):
    """
    Takes the path to a text file and synthesises each sentence it contains
    :param textfile: the path to a text file (string)
    :param args:  the parsed command line argument object giving options
    :return: a list of Audio objects - one for each sentence in order.
    """

    diphone_synth = Synth(wav_folder=args.diphones)
    list_of_outs = []
    with open(textfile) as file:
        for line in file:
            utt = Utterance(line)
            phone_seq = utt.get_phone_seq(reverse=args.reverse)
            out = diphone_synth.synthesise(phone_seq, reverse=args.reverse)
            if args.volume:
                out.rescale(args.volume / 100)

            if args.play:
                out.play()
            list_of_outs.append(out)

    if args.outfile:
        # if extension not specified, set to .wav
        if '.wav' not in args.outfile:
            args.outfile = args.outfile + '.wav'
        audio_instance = simpleaudio.Audio(rate=diphone_synth.rate)
        concat_ary = np.array([], dtype=audio_instance.nptype)
        for element in list_of_outs:
            concat_ary = np.concatenate((concat_ary, element.data))
        audio_instance.data = concat_ary
        audio_instance.save(args.outfile)
    return list_of_outs


def main(args):
    diphone_synth = Synth(wav_folder=args.diphones)

    if args.phrase:
        utt = Utterance(phrase=args.phrase)
        phone_seq = utt.get_phone_seq(reverse=args.reverse)
        out = diphone_synth.synthesise(phone_seq, reverse=args.reverse,
                                       smooth_concat=args.crossfade)

        if args.volume:
            out.rescale(args.volume / 100)

        if args.play:
            out.play()

        if args.outfile:
            # add extension if not specified
            if '.wav' not in args.outfile:
                args.outfile = args.outfile + '.wav'
            out.save(args.outfile)

    elif args.fromfile:
        process_file(args.fromfile, args)


if __name__ == "__main__":
    main(process_commandline())
