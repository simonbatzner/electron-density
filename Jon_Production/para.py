import multiprocessing as mp
import numpy as np
import pprint

def doubler(number, pos, output):
    """
    A doubling function that can be used by a process
    """
    result = number * 2
    proc_name = mp.current_process().name
    print('{0} doubled to {1} by: {2}'.format(
        number, result, proc_name))

    output.put([pos, result])


if __name__ == '__main__':
    output = mp.Queue()
    procs = []
    numbers = list(np.random.randint(100, size=10))
    print(numbers, "\n")

    processes = [mp.Process(target=doubler, args=(numbers[i], i, output)) for i in range(len(numbers))]

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    results = [output.get() for p in processes]
    print(results)

    results_dict = {}

    for result in results:
        results_dict.update({result[0]: result[1]})

    pprint.pprint(results_dict, indent=2)

# # Define an output queue
# output = mp.Queue()
#
# # define a example function
# def rand_string(length, pos, output):
#     """ Generates a random string of numbers, lower- and uppercase chars. """
#     rand_str = ''.join(random.choice(
#                         string.ascii_lowercase
#                         + string.ascii_uppercase
#                         + string.digits)
#                    for i in range(length))
#     output.put((pos, rand_str))
#
# # Setup a list of processes that we want to run
# processes = [mp.Process(target=rand_string, args=(5, x, output)) for x in range(4)]
#
# # Run processes
# for p in processes:
#     p.start()
#
# # Exit the completed processes
# for p in processes:
#     p.join()
#
# # Get process results from the output queue
# results = [output.get() for p in processes]
#
# print(results)