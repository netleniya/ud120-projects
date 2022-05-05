import math


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ages = ages.reshape((1, len(ages)))[0]
    net_worths = net_worths.reshape((1, len(ages)))[0]
    predictions = predictions.reshape((1, len(ages)))[0]

    '''The zip() function is used to take an iterable object as a parameter,
    pack the corresponding elements in the object into tuples,
    and then return a list of these tuples.'''

    cleaned_data = zip(ages, net_worths, abs(net_worths-predictions))
    # sort by error size
    cleaned_data = sorted(cleaned_data, key=lambda x: (x[2]))
    # ceil() function returns the integer of the number
    # and counts the number of elements to delete
    cleaned_num = int(-1 * math.ceil(len(cleaned_data) * 0.1))
    # slice
    cleaned_data = cleaned_data[:cleaned_num]

    # your code goes here

    return cleaned_data
