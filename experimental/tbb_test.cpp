#include "tbb/concurrent_hash_map.h" // For concurrent hash map.

tbb::concurrent_hash_map<int, string> dict;
typedef tbb::concurrent_hash_map<int, string>::accessor dictAccessor; // See notes on accessor below.   

print("  - Insert key, method 1:\n");   
dict.insert({1,"k1"});
print("    - 1: k1\n");

print("  - Insert key, method 2:\n");
dict.emplace(2,"k2");
print("    - 2: k2\n");

string result;

{
    print("  - Read an existing key:\n");   
    dictAccessor accessor;
    const auto isFound = dict.find(accessor, 2);
    // The accessor functions as:
    // (a) a fine-grained per-key lock (released when it goes out of scope).
    // (b) a method to read the value.
    // (c) a method to insert or update the value.
    if (isFound == true) {
        print("    - {}: {}\n", accessor->first, accessor->second);
    }
}

{
    print("  - Atomically insert or update a key:\n");  
    dictAccessor accessor;
    const auto itemIsNew = dict.insert(accessor, 4);
    // The accessor functions as:
    // (a) a fine-grained per-key lock (released when it goes out of scope).
    // (b) a method to read the value.
    // (c) a method to insert or update the value.
    if (itemIsNew == true) {
        print("    - Insert.\n");
        accessor->second = "k4";
    }
    else {
        print("    - Update.\n");
        accessor->second = accessor->second + "+update";
    }
    print("    - {}: {}\n", accessor->first, accessor->second);     
}

{
    print("  - Atomically insert or update a key:\n");          
    dictAccessor accessor;
    const auto itemIsNew = dict.insert(accessor, 4);
    // The accessor functions as:
    // (a) a fine-grained per-key lock which is released when it goes out of scope.
    // (b) a method to read the value.
    // (c) a method to insert or update the value.
    if (itemIsNew == true) {
        print("    - Insert.\n");
        accessor->second = "k4";
    }
    else {
        print("    - Update.\n");
        accessor->second = accessor->second + "+update";
    }
    print("    - {}: {}\n", accessor->first, accessor->second);     
}

{
    print("  - Read the final state of the key:\n");            
    dictAccessor accessor;
    const auto isFound = dict.find(accessor, 4);
    print("    - {}: {}\n", accessor->first, accessor->second);
}