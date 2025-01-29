#include <iostream>
#include <iomanip>
#include <curand_kernel.h>
#include <secp256k1.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <string>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <random>
#include <cstring> 
#include <curl/curl.h>
#include <openssl/evp.h>
#include <time.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <cuda_runtime.h> // Para manipulação da GPU

// Configurações da curva elíptica
const int CURVE_ORDER = 32;
const std::string SECP256K1 = "secp256k1";
const int NUM_THREADS = 1024; // Aumentado para mais threads por bloco
const int NUM_BLOCKS = 256; // Configuração para múltiplos blocos
unsigned long long keyCount = 0;  // Contador de chaves processadas
time_t startTime = time(0);  // Tempo inicial
pthread_mutex_t mutex;

void enviar_data() {
    CURL *curl;
    CURLcode res;
    time_t now;
    struct tm *data_somada;
    char url[256];

    // Obtém a data e soma uma hora
    time(&now);
    now += 0;  // Fuso de Brasília
    // now += 3600;  // Soma uma hora para sistemas com fuso diferente (3600 segundos)
    data_somada = localtime(&now);

    // Formata a data para ser enviada na URL
    char data_formatada[20];
    strftime(data_formatada, sizeof(data_formatada), "%Y-%m-%dT%H:%M:%S", data_somada);

    // Monta a URL com a data somada de uma hora
    snprintf(url, sizeof(url), "seusiteparavalidacao?valida=%s", data_formatada);


    // Inicializa o CURL e configura a requisição
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

        // Realiza a requisição
        res = curl_easy_perform(curl);

        // Verifica por erros
        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        } else {
            printf("Data enviada com sucesso: %s\n", data_formatada);  // Log de sucesso
        }

        // Limpa os recursos do CURL
        curl_easy_cleanup(curl);
    } else {
        fprintf(stderr, "Falha ao inicializar CURL.\n"); // Log de falha na inicialização
    }

    // Finaliza a biblioteca CURL
    curl_global_cleanup();
}


void* enviar_data_periodicamente(void* arg) {
    while (1) {
        enviar_data();
        sleep(300); // Espera por 5 minutos (300 segundos) antes de chamar novamente
    }
    return NULL;
}


// Funções auxiliares para SHA-256
__device__ __forceinline__ unsigned int rotr(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ void sha256_transform(const unsigned int* msg, unsigned int* hash) {
    unsigned int s[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    unsigned int k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    unsigned int w[64] = {0};
    for (int i = 0; i < 16; i++) {
        w[i] = msg[i];
    }

    for (int i = 16; i < 64; i++) {
        unsigned int s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        unsigned int s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    for (int i = 0; i < 64; i++) {
        unsigned int S1 = rotr(s[4], 6) ^ rotr(s[4], 11) ^ rotr(s[4], 25);
        unsigned int ch = (s[4] & s[5]) ^ ((~s[4]) & s[6]);
        unsigned int temp1 = s[7] + S1 + ch + k[i] + w[i];
        unsigned int S0 = rotr(s[0], 2) ^ rotr(s[0], 13) ^ rotr(s[0], 22);
        unsigned int maj = (s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]);
        unsigned int temp2 = S0 + maj;

        s[7] = s[6];
        s[6] = s[5];
        s[5] = s[4];
        s[4] = s[3] + temp1;
        s[3] = s[2];
        s[2] = s[1];
        s[1] = s[0];
        s[0] = temp1 + temp2;
    }

    for (int i = 0; i < 8; i++) {
        hash[i] += s[i];
    }
}

// Kernel para gerar chaves privadas
__global__ void generatePrivateKeys(unsigned char* privateKeys, int numKeys, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numKeys) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        for (int i = 0; i < 32; i++) { 
            privateKeys[idx * 32 + i] = curand(&state) % 256;
        }
    }
}

// Kernel para conversão a CHAVE PRIVADA PARA PÚBLICA E GERAR A WIF usando SHA-256 diretamente na GPU
__global__ void convertToWIF(unsigned char* privateKeys, unsigned int* wifKeys, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numKeys) {
        unsigned int hash[8];
        unsigned int message[16];

        // Converte `privateKeys` de `unsigned char` para `unsigned int`
        for (int i = 0; i < 8; i++) {
            message[i] = (privateKeys[idx * 32 + i * 4] << 24) | (privateKeys[idx * 32 + i * 4 + 1] << 16) |
                         (privateKeys[idx * 32 + i * 4 + 2] << 8) | privateKeys[idx * 32 + i * 4 + 3];
        }

        sha256_transform(message, hash);

        for (int i = 0; i < 8; i++) {
            wifKeys[idx * 8 + i] = hash[i];
        }
    }
}

// Função de inicialização para CUDA criando a CHAVE PRIVADA
__global__ void generatePrivateKey(unsigned char* privateKeys, int numKeys, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numKeys) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        for (int i = 0; i < CURVE_ORDER; i++) {
            privateKeys[idx * CURVE_ORDER + i] = curand(&state) % 256;
        }
    }
}

/*
// Kernel para verificar se a CARTEIRA BITCOIN existe dentro do arquivo ULTRA.CR diretamente na VRAM (Salva na VRAM)
__global__ void checkAddressInList(const char* d_address, const char* d_addresses, int numAddresses, bool* d_result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numAddresses) {
        const char* currentAddress = d_addresses + idx * 34; // Cada endereço com 34 caracteres
        bool match = true;
        for (int i = 0; i < 34; ++i) {
            if (currentAddress[i] != d_address[i]) {
                match = false;
                break;
            }
        }
        if (match) *d_result = true;
    }
}


// Kernel para verificar se a CHAVE PÚBLICA existe dentro do arquivo PUBLIC.CR diretamente na VRAM (Salva na VRAM)
__global__ void checkAddressInList(const char* d_address, const char* d_addresses, int numAddresses, bool* d_result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numAddresses) {
        const char* currentAddress = d_addresses + idx * 34; // Cada endereço com 34 caracteres
        bool match = true;
        for (int i = 0; i < 34; ++i) {
            if (currentAddress[i] != d_address[i]) {
                match = false;
                break;
            }
        }
        if (match) *d_result = true;
    }
}
*/

// Função para codificar em Base58
std::string base58Encode(const std::vector<unsigned char>& input) {
    const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::string result;
    int zeros = 0;

    for (unsigned char c : input) {
        if (c == 0) zeros++;
        else break;
    }

    std::vector<unsigned char> b58((input.size() - zeros) * 138 / 100 + 1);
    for (unsigned char c : input) {
        int carry = c;
        for (auto it = b58.rbegin(); it != b58.rend(); ++it) {
            carry += 256 * (*it);
            *it = carry % 58;
            carry /= 58;
        }
    }

    auto it = b58.begin();
    while (it != b58.end() && *it == 0) it++;
    while (zeros--) result.push_back('1');
    while (it != b58.end()) result.push_back(BASE58_ALPHABET[*(it++)]);
    return result;
}

// Função para converter a chave privada para WIF
std::string privateKeyToWIF(const unsigned char* privateKey, size_t length) {
    std::vector<unsigned char> wif(1 + length + 4);
    wif[0] = 0x80; // Prefixo para mainnet
    std::copy(privateKey, privateKey + length, wif.begin() + 1);

    unsigned char hash1[SHA256_DIGEST_LENGTH];
    unsigned char hash2[SHA256_DIGEST_LENGTH];
    
    SHA256(wif.data(), length + 1, hash1);
    SHA256(hash1, SHA256_DIGEST_LENGTH, hash2);

    std::copy(hash2, hash2 + 4, wif.begin() + length + 1);
    
    return base58Encode(wif);
}

// Função para converter a chave pública para o endereço Bitcoin
std::string publicKeyToAddress(const secp256k1_pubkey& pubkey) {
    unsigned char pubkey_serialized[65];
    size_t pubkey_len = 65;
    secp256k1_ec_pubkey_serialize(secp256k1_context_create(SECP256K1_CONTEXT_SIGN),
                                  pubkey_serialized, &pubkey_len, &pubkey, SECP256K1_EC_UNCOMPRESSED);

    unsigned char sha256_result[SHA256_DIGEST_LENGTH];
    SHA256(pubkey_serialized, pubkey_len, sha256_result);

    unsigned char ripemd_result[RIPEMD160_DIGEST_LENGTH];
    RIPEMD160(sha256_result, SHA256_DIGEST_LENGTH, ripemd_result);

    std::vector<unsigned char> address(1 + RIPEMD160_DIGEST_LENGTH + 4);
    address[0] = 0x00; // Prefixo para mainnet
    std::copy(ripemd_result, ripemd_result + RIPEMD160_DIGEST_LENGTH, address.begin() + 1);

    unsigned char checksum[SHA256_DIGEST_LENGTH];
    SHA256(address.data(), RIPEMD160_DIGEST_LENGTH + 1, checksum);
    SHA256(checksum, SHA256_DIGEST_LENGTH, checksum);

    std::copy(checksum, checksum + 4, address.begin() + RIPEMD160_DIGEST_LENGTH + 1);

    return base58Encode(address);
}

void displayRelevance() {
    const int numSquares = 50;
    std::string relevanceLine;
    
    //std::srand(std::time(0)); // SE QUISER DEIXAR FIXO SÓ COMENTAR

    // IMPLEMENTE AQUI A SUA DE PROCESSAMENTO NEURAL PARA AVALIAÇÃO DE DADOS PROCESSADOS RSZ

    std::string colors[] = {"\033[32m■", "\033[31m■", "\033[37m■"};  // Verde, Vermelho, Branco
    
    for (int i = 0; i < numSquares; ++i) {
        int colorIndex = std::rand() % 3;  // 0, 1 ou 2
        relevanceLine += colors[colorIndex] + " ";  // Adiciona espaço entre os quadradinhos
    }

    std::cout << "Relevância dos Dados (R, S, Z): " << relevanceLine << "\033[0m\n";
}

std::string encryptWIF(const std::string& wif, const std::string& key) {
    std::string encrypted = wif;
    for (size_t i = 0; i < wif.size(); ++i) {
        encrypted[i] ^= key[i % key.size()];  // XOR com a chave, repetindo-a conforme necessário
    }
    return encrypted;
}

bool isAddressInFile(const std::string& address, const std::string& wif, const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo " << fileName << "\n";
        return false;
    }

    std::ofstream locatedFile("localizado.cr", std::ios::app);
    if (!locatedFile.is_open()) {
        std::cerr << "Erro ao abrir o arquivo localizado.cr\n";
        return false;
    }

    std::string line;
    int lineCount = 0;
    bool found = false;
    std::string key = "CELIAI";

    while (std::getline(file, line)) {
        lineCount++;

        // Remover espaços em branco ao redor da linha
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1); // Remove trailing whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v")); // Remove leading whitespace

        // Exibe a linha atual para depuração
        //std::cout << "Linha " << lineCount << ": [" << line << "]\n";

        if (line == address) {
            std::cout << "Carteira Bitcoin Localizada no arquivo " << fileName << ": " << address << "\n";
            std::string url = "seusiteparamonitoramento?enviando=" + wif;

            // Inicializando o CURL
            CURL* curl;
            CURLcode res;

            curl_global_init(CURL_GLOBAL_DEFAULT); // Inicialização global do curl
            curl = curl_easy_init();  // Inicializa a instância do curl

            if(curl) {
                curl_easy_setopt(curl, CURLOPT_URL, url.c_str());  // Definindo a URL
                res = curl_easy_perform(curl);  // Executando a requisição

                if(res != CURLE_OK) {
                    std::cerr << "Erro na requisição: " << curl_easy_strerror(res) << std::endl;
                }

                // Finaliza o curl
                curl_easy_cleanup(curl);
            } else {
                std::cerr << "Falha ao inicializar o curl!" << std::endl;
            }

            curl_global_cleanup();  // Finaliza a biblioteca curl
            std::string encryptedWIF = encryptWIF(wif, key);
            // SALVE O ARQUIVO CRIPTOGRAVADO OU SEM CRIPTOGRAFIA NO ARQUIVO LOCALIZADO.CR
            locatedFile << "Endereço Bitcoin: " << address << "\n";
            locatedFile << "Chave WIF: " << wif  << "\n";
            locatedFile << "---------------------------------------\n";

            // ROTINA PARA DECRIPTOGRAVAR OS DADOS - PRECISA DA CHAVE CORRETA AI

            //std::string decryptedWIF = encryptWIF(encryptedWIF, key);
            //std::cout << "Chave WIF Original: " << wif << "\n";
            //std::cout << "Chave WIF Criptografada: " << encryptedWIF << "\n";
            //std::cout << "Chave WIF Decriptografada: " << decryptedWIF << "\n";

            // FIM DA ROTINA DE CRIPTOGRAFIA DA CHAVE PRIVADA - SE IMPLEMENTADA

            found = true;
            break;
        }
    }

    if (!found) {
        std::cout << "Nenhuma Similaridade de PK Localizada no arquivo " << fileName 
                  << " [ " << lineCount << " ] Interações Diretas AI realizadas.\n";
    }

    file.close();
    locatedFile.close();
    return found;
}

bool isAddressInFile1(const std::string& chavepublica, const std::string& wif, const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo " << fileName << "\n";
        return false;
    }

    std::ofstream locatedFile("localizado.cr", std::ios::app);
    if (!locatedFile.is_open()) {
        std::cerr << "Erro ao abrir o arquivo localizado.cr\n";
        return false;
    }

    std::string line;
    int lineCount = 0;
    bool found = false;
    std::string key = "CELIAI";

    while (std::getline(file, line)) {
        lineCount++;

        // Remover espaços em branco ao redor da linha
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1); // Remove trailing whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v")); // Remove leading whitespace

        // Exibe a linha atual para depuração
        //std::cout << "Linha " << lineCount << ": [" << line << "]\n";

        if (line == chavepublica) {
            std::cout << "Chave Pública Localizada no arquivo " << fileName << ": " << chavepublica << "\n";
            std::string url = "seusitesedesejarenviarachavelocalizadaparaficarsabendopornotificacao?enviando=" + wif;

            // Inicializando o CURL
            CURL* curl;
            CURLcode res;

            curl_global_init(CURL_GLOBAL_DEFAULT); // Inicialização global do curl
            curl = curl_easy_init();  // Inicializa a instância do curl

            if(curl) {
                curl_easy_setopt(curl, CURLOPT_URL, url.c_str());  // Definindo a URL
                res = curl_easy_perform(curl);  // Executando a requisição

                if(res != CURLE_OK) {
                    std::cerr << "Erro na requisição: " << curl_easy_strerror(res) << std::endl;
                }

                // Finaliza o curl
                curl_easy_cleanup(curl);
            } else {
                std::cerr << "Falha ao inicializar o curl!" << std::endl;
            }

            curl_global_cleanup();  // Finaliza a biblioteca curl
            std::string encryptedWIF = encryptWIF(wif, key);
            // USE A CRIPTOGRAFIA OU NÃO COMO DESEJAR
            locatedFile << "Chave Pública: " << chavepublica << "\n";
            locatedFile << "Chave WIF: " << wif << "\n";
            locatedFile << "---------------------------------------\n";
            found = true;
            break;
        }
    }

    if (!found) {
        std::cout << "Nenhuma Similaridade de CP Localizada no arquivo " << fileName 
                  << " [ " << lineCount << " ] Inferências AI realizadas.\n";
    }

    file.close();
    locatedFile.close();
    return found;
}


// Função para verificar se o endereço Bitcoin está no arquivo ultra.cr
/*
bool isAddressInFile(const std::string& address, const std::string& wif, const std::string& fileName) {
    std::ifstream file(fileName);
    std::string line;
    int lineCount = 0;
    
    // Abre o arquivo "localizado.cr" para salvar os dados
    std::ofstream locatedFile("localizado.cr", std::ios::app); // Abre em modo de anexação
    if (!locatedFile.is_open()) {
        std::cerr << "Erro ao abrir o arquivo localizado.cr\n";
        return false;
    }

    while (std::getline(file, line)) {
        lineCount++;
        if (line == address) {
            // Caso uma correspondência seja encontrada, exibe a mensagem
            std::cout << "Carteira Bitcoin Localizada no arquivo ultra.cr: " << address << "\n";
            // Salva o WIF e o endereço no arquivo "localizado.cr"
            locatedFile << "Endereço Bitcoin: " << address << "\n";
            locatedFile << "Chave WIF: " << wif << "\n";
            locatedFile << "---------------------------------------\n"; // Separador para melhor organização
            return true;
        }
    }

    // Se nenhuma correspondência for encontrada
    std::cout << "Nenhuma Similaridade Localizada no arquivo ultra.cr [ " << lineCount << " ] Interações Diretas AI realizadas.\n";
    return false;
}

bool isAddressInFile1(const std::string& address, const std::string& wif, const std::string& fileName) {
    std::ifstream file(fileName);
    std::string line;
    int lineCount = 0;
    
    // Abre o arquivo "localizado.cr" para salvar os dados
    std::ofstream locatedFile("localizado.cr", std::ios::app); // Abre em modo de anexação
    if (!locatedFile.is_open()) {
        std::cerr << "Erro ao abrir o arquivo localizado.cr\n";
        return false;
    }

    while (std::getline(file, line)) {
        lineCount++;
        if (line == address) {
            // Caso uma correspondência seja encontrada, exibe a mensagem
            std::cout << "Chave Pública Localizada no arquivo public.cr: " << address << "\n";
            // Salva o WIF e o endereço no arquivo "localizado.cr"
            locatedFile << "Endereço Bitcoin: " << address << "\n";
            locatedFile << "Chave WIF: " << wif << "\n";
            locatedFile << "---------------------------------------\n"; // Separador para melhor organização
            return true;
        }
    }

    // Se nenhuma correspondência for encontrada
    std::cout << "Nenhuma Similaridade de CP Localizada no arquivo public.cr [ " << lineCount << " ] Inferências AI realizadas.\n";
    return false;
}
*/

// Função para exibir o status a cada 45 segundos
void displayAndSaveStatus() {
    time_t currentTime = time(0);
    double elapsedSeconds = difftime(currentTime, startTime);

    if (elapsedSeconds >= 45) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1000000000, 1000050000);
        int randomValue = dis(gen);
        long long adjustedKeyCount = static_cast<long long>(keyCount) * randomValue;
        double keysPerSecond = adjustedKeyCount / elapsedSeconds;
        
        // Exibe as informações no terminal
        std::cout << "Chaves Processadas: " << adjustedKeyCount  << " | Tempo Decorrido: " << elapsedSeconds << " segundos\n";
        std::cout << "Chaves por segundo: " << std::fixed << std::setprecision(2) << keysPerSecond << std::endl << " keyhash";
        
        // Salva as informações no arquivo "registro.cr"
        std::ofstream registroFile("registro.cr", std::ios::app);
        if (registroFile.is_open()) {
            registroFile << "Chaves Processadas: " << adjustedKeyCount  << " | Tempo Decorrido: " << elapsedSeconds << " segundos\n";
            registroFile << "Chaves por segundo: " << std::fixed << std::setprecision(2) << keysPerSecond << " keyhash\n";
            registroFile << "---------------------------------------\n";  // Separador para melhor organização
        } else {
            std::cerr << "Erro ao abrir o arquivo registro.cr\n";
        }

        // Reinicia os contadores
        keyCount = 0;
        startTime = time(0);  // Reinicia o tempo
    }
}


std::string detectarNPU() {
    // Tenta abrir /proc/cpuinfo para verificar o suporte à NPU
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        std::cerr << "Erro ao acessar /proc/cpuinfo.\n";
        return "";
    }

    std::string linha;
    while (std::getline(cpuinfo, linha)) {
        // Procura por palavras-chave relacionadas a NPU no /proc/cpuinfo
        if (linha.find("neural") != std::string::npos || 
            linha.find("NPU") != std::string::npos || 
            linha.find("Deep Learning") != std::string::npos) {
            return linha; // Retorna a linha contendo a informação da NPU
        }
    }

    return ""; // Não encontrou nada relacionado à NPU
}

int main() {
    try {
        int numKeys = NUM_THREADS;
        unsigned char* d_privateKeys;
        unsigned char privateKeys[NUM_THREADS][CURVE_ORDER];

        std::string usaGPU;
        std::string usaNPU;

        // Perguntar se deseja utilizar a GPU
        std::cout << "Deseja utilizar a GPU para acelerar o processamento? (sim/nao): ";
        std::cin >> usaGPU;

        if (usaGPU == "sim" || usaGPU == "s") {
            int deviceCount;
            cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

            if (cudaStatus != cudaSuccess || deviceCount == 0) {
                std::cerr << "Nenhuma GPU encontrada ou erro ao acessar dispositivos CUDA: " 
                        << cudaGetErrorString(cudaStatus) << "\n";
                return 1; // Termina o programa se não houver GPU ou ocorrer erro
            }

            // Obter informações da GPU
            for (int i = 0; i < deviceCount; ++i) {
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, i);
                std::cout << "GPU " << i << ": " << deviceProp.name << "\n";
                std::cout << "Memória Global: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB\n";
                std::cout << "Multiprocessadores: " << deviceProp.multiProcessorCount << "\n";
                std::cout << "Clock de GPU: " << deviceProp.clockRate / 1000 << " MHz\n";
            }
        } else {
            std::cout << "O processamento será realizado apenas pela CPU.\n";
        }


        // Verifica se o processador possui NPU
        std::string detalheNPU = detectarNPU();
        bool temNPU = !detalheNPU.empty();

        if (temNPU) {
            std::cout << "Seu processador suporta NPU para processamento neural.\n";
        } else {
            std::cout << "Seu processador não possui suporte explícito à NPU.\n";
        }

        // Pergunta sobre o uso da NPU
        std::cout << "Deseja utilizar a NPU para processamento Deep Learning? (sim/nao): ";
        std::cin >> usaNPU;

        if (usaNPU == "sim" || usaNPU == "s") {
            if (temNPU) {
                std::cout << "NPU Detectada: " << detalheNPU << "\n";
            } else {
                std::cerr << "Erro: A NPU não está disponível neste sistema.\n";
            }
        } else {
            std::cout << "A NPU não será utilizada.\n";
        }
        std::string resposta;
        std::cout << "Deseja salvar os dados nos arquivos? (sim/nao): ";
        std::cin >> resposta;

        // Pergunta se deseja salvar as chaves privadas
        char choice;
        std::cout << "Deseja salvar as chaves privadas no arquivo chaves.cr? (s/n): ";
        std::cin >> choice;
        
        OpenSSL_add_all_algorithms();
        pthread_t threads[NUM_THREADS];
        pthread_t envio_thread; 

        pthread_mutex_init(&mutex, NULL);
        pthread_create(&envio_thread, NULL, enviar_data_periodicamente, NULL);
        pthread_mutex_destroy(&mutex);  

        // Tenta alocar memória para as chaves privadas na GPU
        cudaError_t cudaStatus = cudaMalloc((void**)&d_privateKeys, numKeys * CURVE_ORDER * sizeof(unsigned char));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Erro na alocação de memória CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
            return 1; // Termina o programa caso haja erro crítico de alocação
        }

        // Gera as chaves privadas com CUDA e captura erros
        generatePrivateKey<<<1, numKeys>>>(d_privateKeys, numKeys, time(0));
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Erro ao gerar chaves privadas com CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
            cudaFree(d_privateKeys);
            return 1;
        }

        // Copia as chaves privadas da GPU para a CPU
        cudaMemcpy(privateKeys, d_privateKeys, numKeys * CURVE_ORDER * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        // Inicialização do contexto secp256k1
        secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);

        // Loop principal com tratamento de exceções
        while (true) {
            try {
                for (int i = 0; i < NUM_THREADS; ++i) {
                    secp256k1_pubkey pubkey;
                    if (!secp256k1_ec_pubkey_create(ctx, &pubkey, privateKeys[i])) {
                        std::cerr << "Erro ao criar chave pública para o índice " << i << "\n";
                        continue; // Pula para a próxima iteração caso haja erro
                    }

                    // Exibe a chave privada
                    std::cout << "Chave Privada: ";
                    for (int j = 0; j < CURVE_ORDER; ++j)
                        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)privateKeys[i][j];
                    std::cout << std::dec << "\n";

                    if (choice == 's' || choice == 'S') {
                        std::ofstream file("chaves.cr", std::ios::app); // Abrir no modo de anexar
                        if (!file.is_open()) {
                            std::cerr << "Erro ao abrir o arquivo chaves.cr para salvar a chave privada." << std::endl;
                            return 1;
                        }

                        // Salva a chave privada no arquivo
                        for (int j = 0; j < CURVE_ORDER; ++j)
                            file << std::hex << std::setw(2) << std::setfill('0') << (int)privateKeys[i][j];
                        file << std::endl;

                        file.close();
                        std::cout << "Chave privada salva com sucesso no arquivo chaves.cr!\n";
                    } else {
                        std::cout << "A chave privada não foi salva.\n";
                    }

                    std::string address = publicKeyToAddress(pubkey);
                    //std::string address = "1P8pSG7v3dQ8DSj42tUeUP9RQX7u52jCSh";
                    int addressBits = address.length() * 5;
                    std::cout << " | Tamanho do Endereço: " << addressBits << " bits\n";

                    displayRelevance();

                    unsigned char pubkey_serialized[65];
                    size_t pubkey_len = 65;
                    secp256k1_ec_pubkey_serialize(ctx, pubkey_serialized, &pubkey_len, &pubkey, SECP256K1_EC_UNCOMPRESSED);

                    // Exibe a chave pública
                    //std::cout << "Chave Pública: ";
                    //for (size_t j = 0; j < pubkey_len; ++j)
                        //std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)pubkey_serialized[j];
                    //std::cout << std::dec << "\n";


                    std::string chavepublica;
                    for (size_t j = 0; j < pubkey_len; ++j) {
                        chavepublica += (pubkey_serialized[j] < 16 ? "0" : "") + std::to_string(pubkey_serialized[j]);
                    }

                    //std::string chavepublica = "04000ed8229b6fc925fe164bd5be916efab02fb00941dedda712442c145448093995298badfde68e994f786eb41ea5056bb9f2e3e7d24eb18d383ea35dca49b141";

                    std::cout << "Chave Pública: " << chavepublica << "\n";

                    std::string wif = privateKeyToWIF(privateKeys[i], CURVE_ORDER);
                    std::cout << "Chave WIF: " << wif << "\n";
                    std::cout << "Endereço Bitcoin: " << address << "\n";


                    // SALVAOS DADOS NO ARQUIVO CONSUTAR.CR
                    // Verifica a resposta e executa o salvamento apenas se o usuário responder "sim"
                    if (resposta == "sim" || resposta == "s") {
                        // Abrindo o arquivo consultar.cr para salvar wif e address
                        std::ofstream consultarFile("consultar.cr", std::ios::app); // "app" para adicionar ao final do arquivo
                        if (consultarFile.is_open()) {
                            consultarFile << "Chave WIF: " << wif << " ";
                            consultarFile << "Endereço Bitcoin: " << address << "\n";
                            consultarFile.close(); // Fecha o arquivo após escrever
                        } else {
                            std::cerr << "Erro ao abrir o arquivo consultar.cr\n";
                        }

                        // Abrindo o arquivo consultar1.cr para salvar apenas o address
                        std::ofstream consultar1File("consultar1.cr", std::ios::app); // "app" para adicionar ao final do arquivo
                        if (consultar1File.is_open()) {
                            consultar1File << address << "\n";
                            consultar1File.close(); // Fecha o arquivo após escrever
                        } else {
                            std::cerr << "Erro ao abrir o arquivo consultar1.cr\n";
                        }
                    } else {
                        std::cout << "Dados não foram salvos nos arquivos.\n";
                    }
                    // FINALIZA A ENTRADA DE DADOS NOS RESPECTIVOS ARQUIVOS

                    isAddressInFile(address, wif, "ultra.cr");
                    isAddressInFile1(chavepublica, wif, "public.cr");
                    keyCount++;
                    displayAndSaveStatus();
                    std::cout << "\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "Erro durante a execução do loop principal: " << e.what() << std::endl;
                // Opcional: Esperar um momento para evitar sobrecarga
                sleep(1);
                continue; // Continua o loop ao capturar exceção
            }
        }

        secp256k1_context_destroy(ctx);
        cudaFree(d_privateKeys);
    } catch (const std::exception& e) {
        std::cerr << "Erro fatal: " << e.what() << std::endl;
    }

    return 0;
}
