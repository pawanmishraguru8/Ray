#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <map>
#include <set>
#include <stack>
#include <unordered_map>
#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>

// Define the point structure for 3D points
struct Point {
    float x, y, z;

    bool operator<(const Point& other) const {
        return std::tie(x, y, z) < std::tie(other.x, other.y, other.z);
    }

    bool operator==(const Point& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    Point operator-(const Point& p) const {
        return { x - p.x, y - p.y, z - p.z };
    }

    Point operator+(const Point& p) const {
        return { x + p.x, y + p.y, z + p.z };
    }

    Point operator*(float scalar) const {
        return { x * scalar, y * scalar, z * scalar };
    }

    Point cross(const Point& other) const {
        return { y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x };
    }

    float magnitude() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    // Normalize the vector
    Point normalize() const {
        float length = std::sqrt(x * x + y * y + z * z);
        return { x / length, y / length, z / length };
    }
};

// Define a triangle structure for face adjacency
struct Triangle {
    Point vertices[3];
};

// Cross product of two vectors
Point cross(const Point& p1, const Point& p2) {
    return { p1.y * p2.z - p1.z * p2.y, p1.z * p2.x - p1.x * p2.z, p1.x * p2.y - p1.y * p2.x };
}

// Dot product of two vectors
float dot(const Point& p1, const Point& p2) {
    return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

// Helper function to find a point in a vector
int findPointIndex(const std::vector<Point>& points, const Point& p) {
    auto it = std::find(points.begin(), points.end(), p);
    return (it != points.end()) ? std::distance(points.begin(), it) : -1;
}

// Function to calculate the area of a triangle
float calculateTriangleArea(const Triangle& tri) {
    Point AB = tri.vertices[1] - tri.vertices[0];
    Point AC = tri.vertices[2] - tri.vertices[0];
    return 0.5f * AB.cross(AC).magnitude();
}

// Function to calculate the normal of a triangle
Point calculateNormal(const Triangle& tri) {
    Point v1 = { tri.vertices[1].x - tri.vertices[0].x, tri.vertices[1].y - tri.vertices[0].y, tri.vertices[1].z - tri.vertices[0].z };
    Point v2 = { tri.vertices[2].x - tri.vertices[0].x, tri.vertices[2].y - tri.vertices[0].y, tri.vertices[2].z - tri.vertices[0].z };

    // Cross product to find the normal
    Point normal = {
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    };

    // Normalize the normal
    float length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    normal.x /= length;
    normal.y /= length;
    normal.z /= length;

    return normal;
}

struct Ray {
    Point origin;
    Point direction;
};

const float EPSILON = 1e-6;


std::map<int, std::vector<Triangle>> connectedComponentTriangles;

std::vector<std::vector<Triangle>> getConnectedComponents(const std::string& filename, std::vector<Triangle>& faces) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }

    // Containers for unique vertices, edges, and faces
    std::vector<Point> uniqueVertices;
    std::vector<std::pair<int, int>> edges;
    std::vector<std::vector<int>> edgeToFaces;

    // STL Binary Header Check
    char header[80];
    file.read(header, 80); // Read header

    // Number of triangles (faces)
    unsigned int numTriangles = 0;
    file.read(reinterpret_cast<char*>(&numTriangles), sizeof(numTriangles));

    for (unsigned int i = 0; i < numTriangles; ++i) {
        // Skip the normal vector (3 floats)
        file.ignore(3 * sizeof(float));

        // Read the three vertices of the triangle
        Triangle triangle;
        for (int j = 0; j < 3; ++j) {
            file.read(reinterpret_cast<char*>(&triangle.vertices[j].x), sizeof(float));
            file.read(reinterpret_cast<char*>(&triangle.vertices[j].y), sizeof(float));
            file.read(reinterpret_cast<char*>(&triangle.vertices[j].z), sizeof(float));

            // Add the point to uniqueVertices if not already present
            if (findPointIndex(uniqueVertices, triangle.vertices[j]) == -1) {
                uniqueVertices.push_back(triangle.vertices[j]);
            }
        }

        // Add the triangle to the faces list
        faces.push_back(triangle);

        // Add edges (undirected) and associate with the current face index
        auto addEdge = [&](const Point& p1, const Point& p2) {
            int idx1 = findPointIndex(uniqueVertices, p1);
            int idx2 = findPointIndex(uniqueVertices, p2);
            std::pair<int, int> edge = { std::min(idx1, idx2), std::max(idx1, idx2) };

            // Find or insert the edge
            auto it = std::find(edges.begin(), edges.end(), edge);
            if (it == edges.end()) {
                edges.push_back(edge);
                edgeToFaces.push_back({ static_cast<int>(faces.size()) - 1 });
            }
            else {
                edgeToFaces[std::distance(edges.begin(), it)].push_back(static_cast<int>(faces.size()) - 1);
            }
            };

        addEdge(triangle.vertices[0], triangle.vertices[1]);
        addEdge(triangle.vertices[1], triangle.vertices[2]);
        addEdge(triangle.vertices[2], triangle.vertices[0]);

        // Skip attribute byte count
        file.ignore(2);
    }

    file.close();

    // Build adjacency graph
    std::vector<std::vector<int>> adjacencyGraph(faces.size());
    for (size_t i = 0; i < edges.size(); ++i) {
        const auto& faceIndices = edgeToFaces[i];
        if (faceIndices.size() == 2) { // Shared edge between two faces
            adjacencyGraph[faceIndices[0]].push_back(faceIndices[1]);
            adjacencyGraph[faceIndices[1]].push_back(faceIndices[0]);
        }
    }

    // Find connected components using Iterative DFS
    std::vector<bool> visited(faces.size(), false);
    std::vector<std::vector<Triangle>> connectedComponents;

    for (size_t i = 0; i < adjacencyGraph.size(); ++i) {
        if (!visited[i]) {
            std::vector<Triangle> component;
            std::stack<int> stack;
            stack.push(i);

            while (!stack.empty()) {
                int node = stack.top();
                stack.pop();

                if (!visited[node]) {
                    visited[node] = true;
                    component.push_back(faces[node]);

                    for (int neighbor : adjacencyGraph[node]) {
                        if (!visited[neighbor]) {
                            stack.push(neighbor);
                        }
                    }
                }
            }

            connectedComponents.push_back(component);
        }
    }

    return connectedComponents;
}

// Function to write outer faces to a new binary STL file
void writeSTLBinary(const std::string& filename, const std::map<int, Triangle>& outerFaces) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for writing " << filename << std::endl;
        return;
    }

    // Write the 80-byte header (empty or custom)
    char header[80] = {};
    file.write(reinterpret_cast<const char*>(header), sizeof(header));

    // Write number of triangles
    unsigned int numTriangles = outerFaces.size();
    file.write(reinterpret_cast<const char*>(&numTriangles), sizeof(numTriangles));

    // Write each triangle
    for (const auto& [index, tri] : outerFaces) {
        // Write a normal vector (dummy normal vector for simplicity)
        Point normal = { 0.0f, 0.0f, 0.0f };
        file.write(reinterpret_cast<const char*>(&normal), sizeof(normal));

        // Write triangle vertices
        for (int i = 0; i < 3; ++i) {
            file.write(reinterpret_cast<const char*>(&tri.vertices[i]), sizeof(Point));
        }

        // Write attribute byte count (set to 0)
        unsigned short attributeByteCount = 0;
        file.write(reinterpret_cast<const char*>(&attributeByteCount), sizeof(attributeByteCount));
    }

    file.close();
}


// // Ray-Triangle Intersection Function
// bool rayIntersectsTriangle(const Ray& ray, const Triangle& triangle) {
//     const float EPSILON = 1e-6f;

//     // Edges of the triangle
//     Point edge1 = {
//         triangle.vertices[1].x - triangle.vertices[0].x,
//         triangle.vertices[1].y - triangle.vertices[0].y,
//         triangle.vertices[1].z - triangle.vertices[0].z
//     };
//     Point edge2 = {
//         triangle.vertices[2].x - triangle.vertices[0].x,
//         triangle.vertices[2].y - triangle.vertices[0].y,
//         triangle.vertices[2].z - triangle.vertices[0].z
//     };

//     // Compute the determinant
//     Point h = {
//         ray.direction.y * edge2.z - ray.direction.z * edge2.y,
//         ray.direction.z * edge2.x - ray.direction.x * edge2.z,
//         ray.direction.x * edge2.y - ray.direction.y * edge2.x
//     };

//     float a = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;

//     // If determinant is near zero, the ray lies in the plane of the triangle
//     if (std::abs(a) < EPSILON) {
//         return false;
//     }

//     float f = 1.0f / a;

//     // Calculate distance from vertex[0] to ray origin
//     Point s = {
//         ray.origin.x - triangle.vertices[0].x,
//         ray.origin.y - triangle.vertices[0].y,
//         ray.origin.z - triangle.vertices[0].z
//     };

//     // Calculate u parameter and test bounds
//     float u = f * (s.x * h.x + s.y * h.y + s.z * h.z);
//     if (u < 0.0f || u > 1.0f) {
//         return false;
//     }

//     // Prepare to test v parameter
//     Point q = {
//         s.y * edge1.z - s.z * edge1.y,
//         s.z * edge1.x - s.x * edge1.z,
//         s.x * edge1.y - s.y * edge1.x
//     };

//     // Calculate v parameter and test bounds
//     float v = f * (ray.direction.x * q.x + ray.direction.y * q.y + ray.direction.z * q.z);
//     if (v < 0.0f || u + v > 1.0f) {
//         return false;
//     }

//     // Calculate t, ray intersection distance
//     float t = f * (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z);

//     // Ray intersection occurs if t > EPSILON
//     return t > EPSILON;
// }


// Initialize Embree Device
RTCDevice initializeDevice() {
    RTCDevice device = rtcNewDevice(nullptr);
    if (!device) {
        std::cerr << "Error creating Embree device!" << std::endl;
        exit(1);
    }
    return device;
}

// Build BVH
RTCScene buildBVH(RTCDevice device, const std::vector<Triangle>& triangles) {
    RTCScene scene = rtcNewScene(device);

    RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    auto* vertices = (float*)rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(float) * 3, triangles.size() * 3);
    auto* indices = (unsigned int*)rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(unsigned int) * 3, triangles.size());

    for (size_t i = 0; i < triangles.size(); ++i) {
        // Set vertices
        for (int j = 0; j < 3; ++j) {
            vertices[i * 9 + j * 3 + 0] = triangles[i].vertices[j].x;
            vertices[i * 9 + j * 3 + 1] = triangles[i].vertices[j].y;
            vertices[i * 9 + j * 3 + 2] = triangles[i].vertices[j].z;
        }
        // Set indices
        indices[i * 3 + 0] = i * 3 + 0;
        indices[i * 3 + 1] = i * 3 + 1;
        indices[i * 3 + 2] = i * 3 + 2;
    }

    rtcCommitGeometry(geometry);
    rtcAttachGeometry(scene, geometry);
    rtcReleaseGeometry(geometry);
    rtcCommitScene(scene);

    return scene;
}

// Ray Intersection Function using BVH
bool rayIntersectsScene(const RTCScene& scene, const Ray& ray) {
    struct RTCIntersectContext context;
    rtcInitIntersectContext(&context);

    RTCRayHit rayHit;
    rayHit.ray.org_x = ray.origin.x;
    rayHit.ray.org_y = ray.origin.y;
    rayHit.ray.org_z = ray.origin.z;
    rayHit.ray.dir_x = ray.direction.x;
    rayHit.ray.dir_y = ray.direction.y;
    rayHit.ray.dir_z = ray.direction.z;
    rayHit.ray.tnear = 0.0f;
    rayHit.ray.tfar = std::numeric_limits<float>::infinity();
    rayHit.ray.mask = -1;
    rayHit.ray.flags = 0;
    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(scene, &context, &rayHit);

    return rayHit.hit.geomID != RTC_INVALID_GEOMETRY_ID;
}




int main() {
    const std::string inputFilename = "d:\\pawan\\cpp test\\4recpptest145.stl";
    const std::string outputFilename = "d:\\pawan\\cpp test\\4recpptest146.stl";
    int outerCount = 0;
    int innerCount = 0;
    std::vector<Triangle> faces;  // Fill with actual data
     RTCScene scene = buildBVH(device, faces);

    Ray ray = { {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f} }; // Define a ray
    if (rayIntersectsScene(scene, ray)) {
        std::cout << "Ray intersects the scene!" << std::endl;
    } else {
        std::cout << "No intersection." << std::endl;
    }

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);


    // Get connected components from input file
    auto connectedComponents = getConnectedComponents(inputFilename, faces);
    std::map<int, Triangle> outerFaces;

    std::cout << "Total connected components: " << connectedComponents.size() << std::endl;

    int globalIndex = 0; // To track global face indices for outerFaces
    for (size_t componentIdx = 0; componentIdx < connectedComponents.size(); ++componentIdx) {
        auto& component = connectedComponents[componentIdx];
        bool isOuter = false;
        int intersectionCount = 0;

        // Print all triangles in the current component for debugging
        std::cout << "Component " << componentIdx << " contains " << component.size() << " triangles." << std::endl;

        // Find the largest triangle and its normal
        float largestArea = 0.0f;
        Triangle largestTriangle;
        for (const auto& tri : component) {
            float area = calculateTriangleArea(tri);
            if (area > largestArea) {
                largestArea = area;
                largestTriangle = tri;
            }
        }

        // Calculate ray origin and direction
        Point normal = calculateNormal(largestTriangle);
        Point rayOrigin = {
            (largestTriangle.vertices[0].x + largestTriangle.vertices[1].x + largestTriangle.vertices[2].x) / 3,
            (largestTriangle.vertices[0].y + largestTriangle.vertices[1].y + largestTriangle.vertices[2].y) / 3,
            (largestTriangle.vertices[0].z + largestTriangle.vertices[1].z + largestTriangle.vertices[2].z) / 3
        };

        float offset = 1e-2f;
        float normalLength = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        normal.x /= normalLength;
        normal.y /= normalLength;
        normal.z /= normalLength;

       /* rayOrigin.x += offset;
        rayOrigin.y += offset;
        rayOrigin.z += offset;*/

        // Apply the offset in the direction of the normal vector
        rayOrigin.x += offset * normal.x;
        rayOrigin.y += offset * normal.y;
        rayOrigin.z += offset * normal.z;

        Ray ray = { rayOrigin, normal };

        // Check ray intersections with other components
        for (size_t otherComponentIdx = 0; otherComponentIdx < connectedComponents.size(); ++otherComponentIdx) {
            if (componentIdx == otherComponentIdx) continue;

            for (const auto& tri : connectedComponents[otherComponentIdx]) {
                if (rayIntersectsTriangle(ray, tri)) {
                    std::cout << "Ray from component " << componentIdx << " intersects component " << otherComponentIdx << std::endl;
                    intersectionCount++;
                    break;
                }
            }
        }

        // Determine if the component is outer or inner
        if (intersectionCount % 2 == 0) {
            outerCount++;
            for (const auto& tri : component) {
                outerFaces[globalIndex++] = tri; // Add each triangle to outerFaces
            }
            std::cout << "Component " << componentIdx << " is an outer component." << std::endl;
        }
        else {
            innerCount++;
            std::cout << "Component " << componentIdx << " is an inner component." << std::endl;
        }
    }

    // Print total outer and inner counts
    std::cout << "Total outer components: " << outerCount << std::endl;
    std::cout << "Total inner components: " << innerCount << std::endl;

    // Print all faces in outerFaces for debugging
   /* std::cout << "Outer faces count: " << outerFaces.size() << std::endl;
    for (const auto& [index, tri] : outerFaces) {
        std::cout << "Outer face " << index << ": ("
            << tri.vertices[0].x << ", " << tri.vertices[0].y << ", " << tri.vertices[0].z << ") - ("
            << tri.vertices[1].x << ", " << tri.vertices[1].y << ", " << tri.vertices[1].z << ") - ("
            << tri.vertices[2].x << ", " << tri.vertices[2].y << ", " << tri.vertices[2].z << ")"
            << std::endl;
    }*/

    // Write the outer faces to the STL file
    writeSTLBinary(outputFilename, outerFaces);

    return 0;
}

