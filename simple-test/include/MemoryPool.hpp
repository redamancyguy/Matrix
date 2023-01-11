/*-
 * Copyright (c) 2013 Cosku Acay, http://www.coskuacay.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <climits>
#include <cstddef>

template<std::size_t block_size = 4096>
class MemoryPool {
private:
    struct Block {
        Block *next;
    };
    Block *head;
    Block *tail;
    int last_time_allocated;
    std::map<Block *,std::size_t> real_blocks;
public:
    void add_blocks() {
//        std::cout << "add " << last_time_allocated << "blocks " << std::endl;
        char *pointer = (char *) std::malloc(block_size * last_time_allocated);
        real_blocks[(Block *) pointer] = last_time_allocated;
        auto end = pointer + (block_size * (last_time_allocated - 1));
        for (auto i = (char *) pointer; i < end; i += block_size) {
//            printf(" i pointer  add blocks %p\n",i);
            ((Block *) i)->next = (Block *) (i + block_size);
        }
        tail->next = (Block*)pointer;
        tail = (Block*)end;
    }

    void *allocate() {
//        std::cout << " allocate " << std::endl;
        void *pointer;
        if (head == tail) {
            if (last_time_allocated < (INT32_MAX >> 1)) {
                last_time_allocated = last_time_allocated << 1;
            }
            add_blocks();
        }
        pointer = head;
        head = head->next;
        return pointer;
    }

    void deallocate(void *pointer) {
//        std::cout << " deallocate " << std::endl;
        ((Block *) pointer)->next = head;
        head = (Block *) pointer;
    }

    MemoryPool() {
        real_blocks.clear();
        last_time_allocated = 2;
        head = (Block *) std::malloc(block_size * last_time_allocated);
        real_blocks[head] = last_time_allocated;
        auto end = (Block *) (((char *) head) + (block_size * (last_time_allocated - 1)));
        for (auto i = head; i < end;) {
//            printf(" i pointer %p\n",i);
            i->next = (Block *) (((char *) i) + block_size);
            i = (Block *) (((char *) i) + block_size);
        }
        tail = end;
    }

    ~MemoryPool() {
        for (auto &_: real_blocks) {
//            std::cout<<"free"<<std::endl;
            std::free(_.first);
        }
    }

private:

};

#endif // MEMORY_POOL_H
