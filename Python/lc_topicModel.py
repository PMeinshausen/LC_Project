#!/usr/bin/python

""" 
File: lc_topicModel.py
Class to run topic models for lending club data

Currently only runs on macs because of folder paths.
"""

import csv
import os
import json
import tempfile
import subprocess
import os
import pdb
import sys
from pprint import pprint

class topicModel(object):
    """Builds topic models using Java-based package MALLET"""

    def __init__(self, num_topics=5, 
        MALLET_PATH="/Users/paulmeinshausen/mallet-2.0.7/bin/mallet",
        PRINT_OUTPUT=False):
        """ Constructor create the object to build the topic models in."""
        self._num_topics = num_topics
        self._mallet_path = MALLET_PATH
        self._print_output = PRINT_OUTPUT
        self._id_file_map = {}
        self._RANDOM_SEED = 1
        self._inferencer = None
        self._out_directory = None
        print "Welcome to PyMallet"


    def import_documents(self, input_csv_filename="data.csv"):
        """Takes a csv file and returns a temporary directory
        of individual documents.
        The csv file is expected to be a two-element list,
        where the first element is the column name for the 
        document id, and the 2nd element is the column name
        for the document text.
        """

        _matrix = None

        id_idx = 0
        document_idx = 1

        temp_directory = tempfile.mkdtemp()
        out_directory = tempfile.mkdtemp()

        self._out_directory = out_directory

        with open(input_csv_filename, "rU") as input_csv_file:
            input_reader = csv.reader(
                input_csv_file, delimiter=',', quotechar='"', 
                escapechar='\\')

            for row in input_rader:
                item_id, text = row[id_idx], row[document_idx]
                f = tempfile.NamedTemporaryFile(
                    delete=False, dir=temp_directory, suffix='.txt')
                self._id_file_map[f.name] = item_id
                f.write(text)
                f.close()

            my_env = os.environ.copy()

            OPT_INTERVAL = 10
            INFERENCER_FILE = "/td_T5-inferencer"

            '''Run MALLET on this directory'''
            subprocess.call(
                [self._mallet_path, "train-topics", "--input",
                temp_directory, "--output", self._out_directory + \
                "/mymodel.mallet", "--keep-sequence", "--remove-stopwords"],
                env=my_env)

            print "-- Training topics and creating inferencer."

            subprocess.call(
                [self._mallet_path, "train-topics", "--input",
                self._out_directory + "/mymodel.mallet",
                "--num-topics", str(self._num_topics), "--optimize-interval",
                str(OPT_INTERVAL), "--output-state", 
                self._out_directory + '/mymodel_T5.gz', "--output-topic-keys",
                self._out_directory + '/mymodel_T5-keys.txt', "--output-doc-topics",
                self._out_directory + "/td_T5-composition.txt", "--inferencer-filename",
                self._out_directory + INFERENCER_FILE], env=my_env)

            self._inferencer = self._out_directory + INFERENCER_FILE

            subprocess.call(["ls", "-afgh", temp_directory])
            subprocess.call(["ls", "-afgh", self._out_directory])

            tc_dict = self.convert_topic_composition_to_dict(
                self._out_directory + "/td_T5-composition.txt", self._num_topics)

            matrix = self.create_topic_document_matrix(topic_composition_dict=tc_dict)


            for f in self._id_file_map:
                os.unlink(f)

            self.matrix = matrix

            return {
                "matrix": matrix,
                "topic_lists": None,
                "topic_definitions": None,
            }

    def infer_topic_weights(self, text=""):
        my_env = os.environ.copy()

        if not self._inferencer:
            raise Exception, "No inferencer has been created (%s)" % (self._inferencer,)
        if not self._num_topics:
            raise Exception, "Number of topics is not set (%s)" % (self._num_topics,)
        if not self._out_directory:
            raise Exception, "Out directory isnt set (%s)" % (self._out_directory,)

        tmp_dir = tempfile.mkdtemp()
        f = tempfile.NamedTemporaryFile(
                            delete=False, dir=tmp_dir, suffix='.txt')
        f.write(text)
        f.close()
        
        self._id_file_map[f.name] = 'infer_doc'

        print "-- Preparing data for inferencer"
        subprocess.call([self._mallet_path, "infer-topics",
            "--inferencer", self._inferencer, "--input",
            self._out_directory + "/new_document.mallet",
            "--output-doc-topics", self._out_directory + "/new_document_topics.txt",
            "--random-seed", str(self._RANDOM_SEED)], env=my_env)

        topic_composition_csv = self._out_directory + "/new_document_topics.txt"
        tc_dict = self.convert_topic_composition_to_dict(topic_composition_csv, self._num_topics, delimiter_=" ")
        matrix = self.create_topic_document_matrix(topic_composition_dict=tc_dict)
        return matrix[0][1:]

    def convert_topic_composition_to_dict(self, input_csv_filename='',
        topic_count=0, delimiter_='\t'):
        """Takes a topic composition tab-separted file. Converts it into a dict.
        Each item's key is the file_id (<id>.txt) and value is a dict of mallet_id, file_path,
        and topic_probabilities [list of length n, ordered by topic id 0-to-n]"""

        print "convert_topic_composition_to_dict"

        output = {}
        with open(input_csv_filename, "rU") as input_csv_file:
            input_reader = csv.reader(
                input_csv_file, delimiter=delimiter_, quotechar='"')

            next(input_reader, None)

            for row in input_reader:
                # convert topic probabilities to a list, ordered by topic id (note:
                # we lose probability ordering)
                while True:
                    try:
                        row.remove('')
                    except:
                        break

                probabilities_from_csv = row[2:]

                # topic_count = len(probabilities_from_csv) / 2

                isTopicId = True
                topicId = 0
                topic_probabilities = [0] * topic_count
                for p in probabilities_from_csv:
                    if isTopicId:
                        topicId = int(p)
                    else:  # topic probability
                        topic_probabilities[topicId] = float(p)

                    isTopicId = not isTopicId

                file_name, file_extension = os.path.splitext(row[1])
                file_id = os.path.basename(file_name).strip(file_extension)
                d = {
                    'mallet_id': row[0],
                    'file_path': row[1].strip('file:'),
                    # 'file_id' : file_id,
                    'topic_probabilities': topic_probabilities
                }
                output[file_id] = d

            return output

    def create_topic_document_matrix(self, outfile='', topic_composition_dict={}):
        """Creates a single file which is a topic-document matrix.
        Rows are the documents, columns are the topics, values are the score for the topic in the document."""

        # global self._id_file_map

        matrix = []

        for k in topic_composition_dict:
            row_values = []
            row_values.append(self._id_file_map[ topic_composition_dict[k]['file_path'] ])
            row_values.extend(topic_composition_dict[k]['topic_probabilities'])

            matrix.append(row_values)

        if outfile:
            with open(outfile, 'wb') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"')
                writer.writerows(matrix)

        return matrix

    """ PASS DIRECTORY OF DOCUMENTS TO MALLET/TERMINAL """

    def train_topics(topicnumber, optimize_interval, output_state):
        pass


if __name__ == "__main__":
    p = pymallet()
    p.import_documents()
