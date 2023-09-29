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

        print("faces", self.faces)

        for (face_index, face) in tqdm(enumerate(self.faces)):
            f = [face.a, face.b, face.c]
            f.sort()

            validPairs.add((f[0], f[1]))
            validPairs.add((f[1], f[2]))
            validPairs.add((f[0], f[2]))

        t = 0

        """
        for (vertex1_index, vertex1) in enumerate(self.vertices):
            for (vertex2_index, vertex2) in enumerate(self.vertices[vertex1_index + 1:], start=vertex1_index + 1):
                #print(vertex1, vertex2, vertex1_index, vertex2_index)
                v1 = np.array(vertex1)
                v2 = np.array(vertex2)

                euclidean_distance = np.linalg.norm(v1 - v2)

                if euclidean_distance < t:
                    validPairs.add((vertex1_index, vertex2_index))
        """

        return validPairs

    def getErrors(self, Qs, validPairs):
        errors = []
        vs = []

        for pair in tqdm(validPairs):
            a, b = pair
            Q = Qs[a] + Qs[b]

            dQ = np.array([
                [Q[0,0], Q[0,1], Q[0,2], Q[0,3]],
                [Q[1,0], Q[1,1], Q[1,2], Q[1,3]],
                [Q[2,0], Q[2,1], Q[2,2], Q[2,3]],
                [0,      0,      0,      1]
                ])

            try:
                v = np.linalg.inv(dQ) @ np.array([0, 0, 0, 1]).reshape(-1, 1)
            except np.linalg.LinAlgError:
                """
                @TODO Ajouter les autes cas
                """
                v = ((self.vertices[a] + self.vertices[b]) / 2).reshape(-1, 1)
                v = np.vstack((v, [1]))

            vs.append(v)
            e = (v.T @ Q @ v)[0,0]

            errors.append(e)
        return errors, vs

    def getFacesFromVertice(self, v):
        faces = []

        for face_index,face in enumerate(self.faces):
            if v in [face.a, face.b, face.c]:
                faces.append((face_index,face))
        
        return face

    def getFacesFromVertice(self, v):
        faces = []

        for face_index,face in enumerate(self.faces):
            if v in [face.a, face.b, face.c]:
                faces.append((face_index,face))
        
        return faces

    def contract(self, output):
        """
        Decimates the model stupidly, and write the resulting obja in output.
        """
        operations = []



        # 1 - Compute the Q matrices for all the initial vertices.
        Qs = self.computeQs()
        
        # 2 - Select all valid pairs.
        validPairs = self.getValidPairs()
        errors, vs = self.getErrors(Qs, validPairs)
        
        validPairs = [x for _,x in sorted(zip(errors, validPairs))]
        vs = [x for _,x in sorted(zip(errors, ))]

        # 3 - Compute the optimal contraction target v¯ for each valid pair.
        index = 0
        for index, pair in enumerate(validPairs):
            v1, v2 = pair
            """print(index, validPairs)

            # v1 changé en vs
            operations.append(("ev", v1, self.vertices[v1]))
            self.vertices[v1] = vs[index]

            self.deleted_vertices.add(v2)
            operations.append(('vertex', v2, self.vertices[v2]))

            facesV2 = self.getFacesFromVertice(v2)

            for face_index, face in facesV2:
                if face_index not in self.deleted_faces:
                    if v1 in [face.a, face.b, face.c]:
                        operations.append(('face', face_index, face))
                        self.deleted_faces.add(face_index)
                    else:
                        operations.append(('ef', face_index, face))
                        self.faces[face_index] = obja.Face(v1 if face.a == v2 else face.a,v1 if face.a == v2 else face.b, v1 if face.a == v2 else face.c)

            for index,pair in enumerate(validPairs[index+1:], start=index+1):
                if v2 in pair:
                    validPairs[index] = (pair[0] if pair[0] != v2 else v1, pair[1] if pair[1] != v2 else v1)

        for v_index, v in enumerate(self.vertices):
            if v_index not in self.deleted_vertices:
                operations.append(('vertex', v_index, v))"""

        operations.reverse()

        output_model = obja.Output(output, random_color=True)
        for (ty, index, value) in operations:
            print(ty, index, value) 
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
    np.seterr(invalid='raise')
    model = Decimater()
    model.parse_file('example/square.obj')

    with open('example/square.obja', 'w') as output:
        model.contract(output)


if __name__ == '__main__':
    main()
