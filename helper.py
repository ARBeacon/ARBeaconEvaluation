import numpy as np
from scipy.spatial.transform import Rotation as R
import json

class Orientation:
    def __init__(self, euler):
        self.quat = R.from_euler('xyz', euler).as_quat()
    def getEuler(self):
        return R.from_quat(self.quat).as_euler('xyz')
    @staticmethod
    def avg(orientations):
        quaternions = np.array([obj.quat for obj in orientations])
        quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
        M = np.dot(quaternions.T, quaternions) / len(orientations)
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        avg_quaternion = eigenvectors[:, eigenvalues.argmax()]
        if avg_quaternion[0] < 0:
            avg_quaternion = -avg_quaternion
        return Orientation(R.from_quat(avg_quaternion).as_euler('xyz'))
    def __str__(self) -> str:
        return str(self.getEuler())
    def __repr__(self) -> str:
        return "Orientation: "+self.__str__()

    @staticmethod
    def default():
        return Orientation([0, 0, 0])

    # angle differences in radian
    def distance(self, orientation):
        q1 = self.quat / np.linalg.norm(self.quat)
        q2 = orientation.quat / np.linalg.norm(orientation.quat)
        dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
        angle_difference = 2 * np.arccos(dot_product)
        return angle_difference

class Position:
    def __init__(self, position):
        self.value = np.array(position)
    def __str__(self) -> str:
        return str(self.value)
    def __repr__(self) -> str:
        return "Position: "+self.__str__()
    def distance(self, position):
        return np.linalg.norm(self.value - position.value)
    @staticmethod
    def avg(positions):
        positions = np.array([ position.value for position in positions])
        return Position(np.mean(positions, axis=0))
    @staticmethod
    def default():
        return Position([0, 0, 0])

class Scale:
    def __init__(self, scale):
        self.value = np.array(scale)
    def __str__(self) -> str:
        return str(self.value)
    def __repr__(self) -> str:
        return "Scale: "+self.__str__()
    def distance(self, scale):
        return np.linalg.norm(self.value - scale.value)
    @staticmethod
    def avg(scales):
        scales = np.array([ scale.value for scale in scales])
        return Scale(np.mean(scales, axis=0))
    @staticmethod
    def default():
        return Scale([1, 1, 1])

class ThreeDObject:
    def __init__(self, matrix=None, position=None, orientation=None, scale=None):
        if matrix is not None:
            if matrix.shape != (4, 4):
                raise ValueError("Matrix must be a 4x4 homogeneous matrix.")
            self.matrix = matrix
        else:
            # Default values
            position = position if position is not None else Position.default()
            orientation = orientation if orientation is not None else Orientation.default()
            scale = scale if scale is not None else Scale.default()
            self.matrix = self.create_transformation_matrix(position, orientation, scale)

    @staticmethod
    def create_transformation_matrix(position, orientation, scale):
        [roll, pitch, yaw] = orientation.getEuler()

        # Create individual transformation matrices
        T = np.eye(4)
        T[0:3, 3] = position.value

        R_x = np.array([[1, 0, 0, 0],
                        [0, np.cos(roll), -np.sin(roll), 0],
                        [0, np.sin(roll), np.cos(roll), 0],
                        [0, 0, 0, 1]])

        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch), 0],
                        [0, 1, 0, 0],
                        [-np.sin(pitch), 0, np.cos(pitch), 0],
                        [0, 0, 0, 1]])

        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                        [np.sin(yaw), np.cos(yaw), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        S = np.diag([*scale.value, 1])

        # Combined transformation
        R = R_z @ R_y @ R_x
        transformation_matrix = T @ R @ S
        return transformation_matrix

    def decompose(self):
        scale = Scale(np.linalg.norm(self.matrix[0:3, 0:3], axis=0))

        rotation_matrix = self.matrix[0:3, 0:3] / scale.value

        position = Position(self.matrix[0:3, 3])

        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        pitch = np.arcsin(-rotation_matrix[2, 0])
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        orientation = Orientation(np.array([roll, pitch, yaw]))

        return position, orientation, scale

    def __str__(self) -> str:
        pos, ori, scl = self.decompose()
        return "Metrix: " + str(self.matrix) + "\n" +\
        "Decomposed Position: "+ pos.__str__() + "\n" +\
        "Decomposed Orientation (radians): "+ ori.__str__() + "\n" +\
        "Decomposed Scale: "+ scl.__str__() + "\n"
    def __repr__(self) -> str:
        return "ThreeDObject: \n"+self.__str__()

    @staticmethod
    def avg(objects):
        decomposed = [obj.decompose() for obj in objects]
        positions = [obj[0] for obj in decomposed]
        orientations = [obj[1] for obj in decomposed]
        scales = [obj[2] for obj in decomposed]
        return ThreeDObject(position=Position.avg(positions), orientation=Orientation.avg(orientations), scale=Scale.avg(scales))
    def distance(self, object):
        p1, o1, s1 = self.decompose()
        p2, o2, s2 = object.decompose()
        return {
            "position": p1.distance(p2),
            "orientation": o1.distance(o2),
            "scale": s1.distance(s2)
        }

    def relativePoseRefBy(self, parent):
        relative_matrix = np.linalg.inv(parent.matrix) @ self.matrix
        return ThreeDObject(matrix=relative_matrix)

    def resolveRelativePose(self, relative):
        absolute_matrix = self.matrix @ relative.matrix
        return ThreeDObject(matrix=absolute_matrix)
    
class Scan:
    def __init__(self, entries):
        self.entries = entries
    
    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return Scan(data)

    def firstObjectDelay(self):
        appLaunchTimestamp = next(entry for entry in self.entries if entry["label"]=="Application launched")["timestamp"]
        firstBunnyTimestap = next(entry for entry in self.entries if entry["label"].startswith("Bunny didAdd"))["timestamp"]
        return firstBunnyTimestap-appLaunchTimestamp
    def avgObjectPose(self):
        ImageRefs = dict()
        Bunnies = []
        for entry in self.entries:
            if entry["label"].startswith("Bunny"):
                transform = np.array(entry["content"]["transform"]["columns"]).transpose()
                Bunnies.append(ThreeDObject(matrix=transform))
            elif entry["label"].startswith("ARImageAnchor"):
              transform = np.array(entry["content"]["transform"]["columns"]).transpose()
              name = entry["content"]["name"]
              if name not in ImageRefs:
                  ImageRefs[name] = []
              ImageRefs[name].append(ThreeDObject(matrix=transform))
        avgImageRefs = dict()
        for name, objects in ImageRefs.items():
            avgImageRefs[name] = ThreeDObject.avg(objects)
        return {
            "bunny": ThreeDObject.avg(Bunnies),
            "imagesRefs": avgImageRefs
        }

class ScanGroup:
    def __init__(self, firstScan):
        self.firstScan = firstScan
        self.scanset = dict()  # environmentID: [scan]

    def addScans(self, environmentID, scans):
        if environmentID not in self.scanset:
            self.scanset[environmentID] = []
        self.scanset[environmentID].extend(scans)

    def getFirstObjectDelays(self):
        return {
            environmentID: np.array([scan.firstObjectDelay() for scan in scans])
            for environmentID, scans in self.scanset.items()
        }

    def getAvgObjectPoses(self):
        return {
            environmentID: [scan.avgObjectPose() for scan in scans]
            for environmentID, scans in self.scanset.items()
        }

    def calculateRelativePose(self):
        imageRefs = self.firstScan.avgObjectPose()["imagesRefs"]
        bunny = self.firstScan.avgObjectPose()["bunny"]
        avgImageRef = ThreeDObject.avg(imageRefs.values())
        relative_original = bunny.relativePoseRefBy(avgImageRef)

        relative_targets = dict()

        for environmentID, scans in self.scanset.items():
            relative_targets[environmentID] = []
            for scan in scans:
                avgObjectPose = scan.avgObjectPose()
                bunny = avgObjectPose["bunny"]
                avgImageRef = ThreeDObject.avg([avgObjectPose["imagesRefs"][key] for key in imageRefs.keys()])
                relative = bunny.relativePoseRefBy(avgImageRef)
                relative_targets[environmentID].append(relative)

        return relative_original, relative_targets

    def calculateDistance(self):
        relative_original, relative_targets = self.calculateRelativePose()
        distances = dict()
        for environmentID, relatives in relative_targets.items():
            distances[environmentID] = [relative.distance(relative_original) for relative in relatives]
        return distances