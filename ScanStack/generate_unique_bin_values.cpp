#include <bits/stdc++.h>
using namespace std;

typedef unsigned long long ull;
typedef long long ll;
typedef long double ld;
#define endl '\n'
const int N = 1e7 + 10;
const int M = 1e9 + 7;

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    random_device rd;
    mt19937 gen(rd());
    // uniform_int_distribution<uint32_t> dis(0, 1e9);

    // unordered_set<uint32_t> unique_numbers;
    uniform_int_distribution<int> dis(0, 1e9);

    unordered_set<int> unique_numbers;

    int num_values = 1e8;

    ofstream outfile("unique_numbers_1e8.bin", ios::binary);

    while (unique_numbers.size() < num_values)
    {
        // uint32_t num = dis(gen);
        int num = dis(gen);
        if (unique_numbers.find(num) == unique_numbers.end())
        {
            unique_numbers.insert(num);
            outfile.write(reinterpret_cast<char *>(&num), sizeof(num));
        }
    }

    outfile.close();

    cout << "Generated " << num_values << " unique integers and saved to 'unique_numbers.bin'." << endl;

    return 0;
}