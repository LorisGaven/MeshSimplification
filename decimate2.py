#!/usr/bin/env python

import obja
import numpy as np
import sys
from tqdm import tqdm

class Decimater(obja.Model):
    """
    A simple class that decimates a 3D model stupidly.
    """

    def __init__(self):
        super().__init__()
        self.deleted_faces = set()
        self.deleted_vertices = set()

    def computeQs(self):
        Qs = [np.zeros([4, 4]) for vertex in self.vertices]

        for (face_index, face) in enumerate(self.faces):

            x1, y1, z1 = self.vertices[face.a]
            x2, y2, z2 = self.vertices[face.b]
            x3, y3, z3 = self.vertices[face.c]

            AB = np.array([x2 - x1, y2 - y1, z2 - z1])
            AC = np.array([x3 - x1, y3 - y1, z3 - z1])

            N = np.cross(AB, AC)  # Normal à AB et AC
            N = N / np.linalg.norm(N)
            a, b, c = N
            d = -a * x1 - b * y1 - c * z1

            p = np.array([a, b, c, d]).reshape(-1, 1)

            Kp = p @ p.T

            Qs[face.a] += Kp
            Qs[face.b] += Kp
            Qs[face.c] += Kp

        return Qs

    def getValidPairs(self):

        validPairs = set()

        for (face_index, face) in tqdm(enumerate(self.faces)):
            f = [face.a, face.b, face.c]
            f.sort()

            validPairs.add((f[0], f[1]))
            validPairs.add((f[1], f[2]))
            validPairs.add((f[0], f[2]))

        t = 0

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

    def getError(self, Qs, pair):
        a, b = pair
        Q = Qs[a] + Qs[b]
        
        dQ = np.array([
            [Q[0, 0], Q[0, 1], Q[0, 2], Q[0, 3]],
            [Q[1, 0], Q[1, 1], Q[1, 2], Q[1, 3]],
            [Q[2, 0], Q[2, 1], Q[2, 2], Q[2, 3]],
            [0,      0,      0,      1]
        ])

        try:
            dQinv = np.linalg.inv(dQ)
            v = dQinv @ (np.array([0, 0, 0, 1]).reshape(-1, 1))
        except np.linalg.LinAlgError:
            v = ((np.array(self.vertices[a]) + np.array(self.vertices[b])) / 2).reshape(-1, 1)
            v = np.vstack((v, [1]))

        e = (v.T @ Q @ v)[0, 0]

        return v, e

    def getErrors(self, Qs, validPairs):
        errors = []
        vs = []

        for pair in tqdm(validPairs):
            v, e = self.getError(Qs, pair)

            vs.append(v)
            errors.append(e)
            
        return errors, vs

    def sorteByError(self, l, errors):
        return [x for _, x in sorted(zip(errors, l), key=lambda x: x[0])]

    def contract(self, output):
        """
        Decimates the model stupidly, and write the resulting obja in output.
        """
        operations = []
        operations_face = []

        print("Calcul Qs")
        Qs = self.computeQs()

        print("Calcul validPairs")
        validPairs = self.getValidPairs()

        print("Calcul errors")
        errors, vs = self.getErrors(Qs, validPairs)

        validPairs = self.sorteByError(validPairs, errors)
        vs = self.sorteByError(vs, errors)
        errors = self.sorteByError(errors, errors)


        progress_bar = tqdm(total=len(validPairs), desc="Processing")
        while len(validPairs) != 0:
            v1, v2 = validPairs.pop(0)
            if v1 in self.deleted_vertices or v2 in self.deleted_vertices:
                print("AHHHHHHHHHH")
            v_bar = vs.pop(0)
            errors.pop(0)

            # Remplacer les v2 en v1 + recalculer l'erreur
            for index, pair in enumerate(validPairs):
                if v2 in pair:
                    if v2 == pair[0]:
                        validPairs[index] = (v1, pair[1]) if pair[1] > v1 else (pair[1], v1)
                    elif v2 == pair[1]:
                        validPairs[index] = (v1, pair[0]) if pair[0] > v1 else (pair[0], v1)                    

                    # Calcul de l'erreur
                    vs[index], errors[index] = self.getError(Qs, validPairs[index])      
            
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
            
            # Mise à jour de Qs
            Qs[v1] = Qs[v1] + Qs[v2]

            operations.append(('vertex', v2, self.vertices[v2].copy()))
            self.deleted_vertices.add(v2)

            # Editer v1 -> v_bar
            operations.append(("ev", v1, self.vertices[v1]))
            self.vertices[v1] = v_bar[:-1].flatten().tolist()

            # Enlever les pairs en doubles
            validPairs_ = []
            validPairs_set = set()
            vs_ = []
            errors_ = []

            for index, pair in enumerate(validPairs):
                if pair not in validPairs_set:
                    validPairs_set.add(pair)
                    validPairs_.append(pair)
                    vs_.append(vs[index])
                    errors_.append(errors[index])

            validPairs = validPairs_
            vs = vs_
            errors = errors_

            # Re-tri de validPairs et vs selon errors
            validPairs = self.sorteByError(validPairs, errors)
            vs = self.sorteByError(vs, errors)
            errors = self.sorteByError(errors, errors)

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
