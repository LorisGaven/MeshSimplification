#!/usr/bin/env python

import obja
import numpy as np
import sys
from tqdm import tqdm
import random

class Decimater(obja.Model):
    """
    A simple class that decimates a 3D model stupidly.
    """

    def __init__(self):
        super().__init__()
        self.deleted_faces = set()
        self.deleted_vertices = set()


    def getValidPairs(self):

        validPairs = set()

        for (face_index, face) in tqdm(enumerate(self.faces)):
            f = [face.a, face.b, face.c]
            f.sort()
            validPairs.add((f[0], f[1]))
            validPairs.add((f[1], f[2]))
            validPairs.add((f[0], f[2]))

        t = 0.1

        if t > 0:
            for (vertex1_index, vertex1) in tqdm(enumerate(self.vertices), total=len(self.vertices)):
                for (vertex2_index, vertex2) in enumerate(self.vertices[vertex1_index + 1:], start=vertex1_index + 1):
                    #print(vertex1, vertex2, vertex1_index, vertex2_index)
                    v1 = np.array(vertex1)
                    v2 = np.array(vertex2)

                    euclidean_distance = np.linalg.norm(v1 - v2)

                    if euclidean_distance < t:
                        validPairs.add((vertex1_index, vertex2_index))


        return list(validPairs)
    
    def getDistance(self, v1, v2, faces):
        c1 = self.getCourbure(v1, faces)
        c2 = self.getCourbure(v2, faces)
        
        return c1 + c2

    def getCourbure(self, v, faces):
        cov = self.getMatriceCovariance(v, faces)

        valeurs_propres, vecteurs_propres = np.linalg.eig(cov)
        indices_tries = np.argsort(valeurs_propres)[::-1]
        valeurs_propres = valeurs_propres[indices_tries]

        c = valeurs_propres[0] + valeurs_propres[1]
        c /= 2

        return c
    
    def sorteByDistance(self, l, distances):
        return [x for _, x in sorted(zip(distances, l), key=lambda x: x[0])]

    def computeNormalFace(self, face):
        x1, y1, z1 = self.vertices[face.a]
        x2, y2, z2 = self.vertices[face.b]
        x3, y3, z3 = self.vertices[face.c]

        AB = np.array([x2 - x1, y2 - y1, z2 - z1])
        AC = np.array([x3 - x1, y3 - y1, z3 - z1])

        N = np.cross(AC, AB) 

        norm = np.linalg.norm(N)
        
        return N

    def computeNormalVertice(self, vertice, faces):
        N = np.zeros(3)
        for face in faces[vertice]:
            if face not in self.deleted_faces:
                N += self.computeNormalFace(face)

        norm = np.linalg.norm(N)
        if norm > 0:
            N /= norm
            return N
        return N

    def getFaces(self):
        faces = {k : [] for k in range(len(self.vertices))}
        for face in self.faces:
            if face not in self.deleted_faces:
                faces[face.a] += [face]
                faces[face.b] += [face]
                faces[face.c] += [face]

        return faces

    def getMatriceCovariance(self, vertice, faces):
        face_voisine = [face for face in faces[vertice] if face not in self.deleted_faces]
        Ns = np.zeros((3, 3))

        N_v = self.computeNormalVertice(vertice, faces)

        for i, face in enumerate(face_voisine):
            N = self.computeNormalFace(face) - N_v
            Ns += N @ N.T

        return Ns


    def contract(self, output):
        """
        Decimates the model stupidly, and write the resulting obja in output.
        """
        operations = []
        operations_face = []

        print("Calcul validPairs")
        validPairs = self.getValidPairs()
        progress_bar = tqdm(total=len(validPairs), desc="Processing")
        
        while len(validPairs) != 0:
            #Remove first key of validPairs and assign it to v1 v2
            #Get list of the keys of validPairs
            faces = self.getFaces()
            validPairsDist = [self.getDistance(pair[0], pair[1], faces) for pair in validPairs]
            validPairs = self.sorteByDistance(validPairs, validPairsDist)

            key = validPairs.pop(0)
            v1 = key[0]
            v2 = key[1]
            v_bar = (np.array(self.vertices[v1]) + np.array(self.vertices[v2])) / 2

            # Remplacer les v2 en v1 + recalculer l'erreur
            for index, pair in enumerate(validPairs):
                if v2 in pair:
                    if v2 == pair[0]:
                        validPairs[index] = (v1, pair[1]) if pair[1] > v1 else (pair[1], v1)
                    elif v2 == pair[1]:
                        validPairs[index] = (v1, pair[0]) if pair[0] > v1 else (pair[0], v1)                    
    
            
            for index_face, face in enumerate(self.faces):
                if index_face not in self.deleted_faces:
                    if v1 in [face.a, face.b, face.c] and v2 in [face.a, face.b, face.c]:
                        # Supprimer les faces avec v1v2
                        operations.append(('face', index_face, face.clone()))
                        self.deleted_faces.add(index_face)
                    elif v2 in [face.a, face.b, face.c]:
                        # Changer les faces v2ab -> v1ab
                        operations.append(('ef', index_face, face.clone()))
                        face.a, face.b, face.c = [v if v != v2 else v1 for v in [face.a, face.b, face.c]]
            

            operations.append(('vertex', v2, self.vertices[v2].copy()))
            self.deleted_vertices.add(v2)

            # Editer v1 -> v_bar
            operations.append(("ev", v1, self.vertices[v1]))
            self.vertices[v1] = v_bar.flatten().tolist()

            # Enlever les pairs en doubles
            validPairs_ = []
            validPairs_set = set()

            for index, pair in enumerate(validPairs):
                if pair not in validPairs_set:
                    validPairs_set.add((pair[0], pair[1]))
                    validPairs_.append((pair[0], pair[1]))
   

            validPairs = validPairs_

            progress_bar.update(1)
            progress_bar.total = len(validPairs)

        for face_index, face in enumerate(self.faces):
            if face_index not in self.deleted_faces:
                operations.append(('face', face_index, face.clone()))

        for vertex_index, v in enumerate(self.vertices):
            if vertex_index not in self.deleted_vertices:
                operations.append(('vertex', vertex_index, v))     

        operations.reverse()

        output_model = obja.Output(output, random_color=False)
        for (ty, index, value) in operations:
            if ty == "vertex":
                output_model.add_vertex(index, value)
            elif ty == "face":
                output_model.add_face(index, value)
            elif ty == "ev":
                output_model.edit_vertex(index, value)
            elif ty == "ef":
                output_model.edit_face(index, value)


def main():
    """
    Runs the program on the model given as parameter.
    """

    if len(sys.argv) != 2:
        print("Utilisation : python decimate2.py model")
        exit(0)

    model_name = sys.argv[1]
    np.seterr(invalid='raise')
    model = Decimater()
    model.parse_file(f'example/{model_name}.obj')

    with open(f'example/{model_name}.obja', 'w') as output:
        model.contract(output)


if __name__ == '__main__':
    main()
